# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import inspect
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_all_tasks_by_type
from nv_ingest_api.internal.schemas.meta.udf import UDFStageSchema
from nv_ingest_api.util.imports.callable_signatures import ingest_callable_signature

logger = logging.getLogger(__name__)


@dataclass
class CachedUDF:
    """Cached UDF function with metadata"""

    function: callable
    function_name: str
    signature_validated: bool
    created_at: float
    last_used: float
    use_count: int


class UDFCache:
    """LRU cache for compiled and validated UDF functions"""

    def __init__(self, max_size: int = 128, ttl_seconds: Optional[int] = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CachedUDF] = {}
        self.access_order: List[str] = []  # For LRU tracking

    def _generate_cache_key(self, udf_function_str: str, udf_function_name: str) -> str:
        """Generate cache key from UDF string and function name"""
        content = f"{udf_function_str.strip()}:{udf_function_name}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _evict_lru(self):
        """Remove least recently used item"""
        if self.access_order:
            lru_key = self.access_order.pop(0)
            self.cache.pop(lru_key, None)

    def _cleanup_expired(self):
        """Remove expired entries if TTL is configured"""
        if not self.ttl_seconds:
            return

        current_time = time.time()
        expired_keys = [
            key for key, cached_udf in self.cache.items() if current_time - cached_udf.created_at > self.ttl_seconds
        ]

        for key in expired_keys:
            self.cache.pop(key, None)
            if key in self.access_order:
                self.access_order.remove(key)

    def get(self, udf_function_str: str, udf_function_name: str) -> Optional[CachedUDF]:
        """Get cached UDF function if available"""
        self._cleanup_expired()

        cache_key = self._generate_cache_key(udf_function_str, udf_function_name)

        if cache_key in self.cache:
            # Update access tracking
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)

            # Update usage stats
            cached_udf = self.cache[cache_key]
            cached_udf.last_used = time.time()
            cached_udf.use_count += 1

            return cached_udf

        return None

    def put(
        self, udf_function_str: str, udf_function_name: str, function: callable, signature_validated: bool = True
    ) -> str:
        """Cache a compiled and validated UDF function"""
        cache_key = self._generate_cache_key(udf_function_str, udf_function_name)

        # Evict LRU if at capacity
        while len(self.cache) >= self.max_size:
            self._evict_lru()

        current_time = time.time()
        cached_udf = CachedUDF(
            function=function,
            function_name=udf_function_name,
            signature_validated=signature_validated,
            created_at=current_time,
            last_used=current_time,
            use_count=1,
        )

        self.cache[cache_key] = cached_udf
        self.access_order.append(cache_key)

        return cache_key

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_uses = sum(udf.use_count for udf in self.cache.values())
        most_used = max(self.cache.values(), key=lambda x: x.use_count, default=None)
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_uses": total_uses,
            "most_used_function": most_used.function_name if most_used else None,
            "most_used_count": most_used.use_count if most_used else 0,
        }


# Global cache instance
_udf_cache = UDFCache(max_size=128, ttl_seconds=3600)


def compile_and_validate_udf(udf_function_str: str, udf_function_name: str, task_num: int) -> callable:
    """Compile and validate UDF function (extracted for caching)"""
    # Execute the UDF function string in a controlled namespace
    namespace: Dict[str, Any] = {}
    try:
        exec(udf_function_str, namespace)
    except Exception as e:
        raise ValueError(f"UDF task {task_num} failed to execute: {str(e)}")

    # Extract the specified function from the namespace
    if udf_function_name in namespace and callable(namespace[udf_function_name]):
        udf_function = namespace[udf_function_name]
    else:
        raise ValueError(f"UDF task {task_num}: Specified UDF function '{udf_function_name}' not found or not callable")

    # Validate the UDF function signature
    try:
        ingest_callable_signature(inspect.signature(udf_function))
    except Exception as e:
        raise ValueError(f"UDF task {task_num} has invalid function signature: {str(e)}")

    return udf_function


def get_udf_cache_stats() -> Dict[str, Any]:
    """Get UDF cache performance statistics"""
    return _udf_cache.get_stats()


def udf_stage_callable_fn(control_message: IngestControlMessage, stage_config: UDFStageSchema) -> IngestControlMessage:
    """
    UDF stage callable function that processes UDF tasks in a control message.

    This function extracts all UDF tasks from the control message and executes them sequentially.

    Parameters
    ----------
    control_message : IngestControlMessage
        The control message containing UDF tasks to process
    stage_config : UDFStageSchema
        Configuration for the UDF stage

    Returns
    -------
    IngestControlMessage
        The control message after processing all UDF tasks
    """
    logger.debug("Starting UDF stage processing")

    # Extract all UDF tasks from control message using free function
    try:
        all_task_configs = remove_all_tasks_by_type(control_message, "udf")
    except ValueError:
        # No UDF tasks found
        if stage_config.ignore_empty_udf:
            logger.debug("No UDF tasks found, ignoring as configured")
            return control_message
        else:
            raise ValueError("No UDF tasks found in control message")

    # Process each UDF task sequentially
    for task_num, task_config in enumerate(all_task_configs, 1):
        logger.debug(f"Processing UDF task {task_num} of {len(all_task_configs)}")

        # Get UDF function string and function name from task properties
        udf_function_str = task_config.get("udf_function", "").strip()
        udf_function_name = task_config.get("udf_function_name", "").strip()

        # Skip empty UDF functions if configured to ignore them
        if not udf_function_str:
            if stage_config.ignore_empty_udf:
                logger.debug(f"UDF task {task_num} has empty function, skipping as configured")
                continue
            else:
                raise ValueError(f"UDF task {task_num} has empty function string")

        # Validate that function name is provided
        if not udf_function_name:
            raise ValueError(f"UDF task {task_num} missing required 'udf_function_name' property")

        # Check if UDF function is cached
        cached_udf = _udf_cache.get(udf_function_str, udf_function_name)
        if cached_udf:
            udf_function = cached_udf.function
        else:
            # Compile and validate UDF function
            udf_function = compile_and_validate_udf(udf_function_str, udf_function_name, task_num)
            # Cache the compiled UDF function
            _udf_cache.put(udf_function_str, udf_function_name, udf_function)

        # Execute the UDF function with the control message
        try:
            control_message = udf_function(control_message)
        except Exception as e:
            raise ValueError(f"UDF task {task_num} execution failed: {str(e)}")

        # Validate that the UDF function returned an IngestControlMessage
        if not isinstance(control_message, IngestControlMessage):
            raise ValueError(f"UDF task {task_num} must return an IngestControlMessage, got {type(control_message)}")

        logger.debug(f"UDF task {task_num} completed successfully")

    logger.debug(f"UDF stage processing completed. Processed {len(all_task_configs)} UDF tasks")
    return control_message
