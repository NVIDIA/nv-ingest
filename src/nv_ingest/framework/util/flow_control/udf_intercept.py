# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import hashlib
import inspect
import logging
import os
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_all_tasks_by_type
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


def execute_targeted_udfs(
    control_message: IngestControlMessage, stage_name: str, directive: str
) -> IngestControlMessage:
    """Execute UDFs that target this stage with the given directive."""
    # Early exit if no UDF tasks exist - check by task type, not task ID
    udf_tasks_exist = any(task.type == "udf" for task in control_message.get_tasks())
    if not udf_tasks_exist:
        return control_message

    # Remove all UDF tasks and get them - handle case where no tasks found
    try:
        all_udf_tasks = remove_all_tasks_by_type(control_message, "udf")
    except ValueError:
        # No UDF tasks found - this can happen due to race conditions
        logger.debug(f"No UDF tasks found for stage '{stage_name}' directive '{directive}'")
        return control_message

    # Execute applicable UDFs and collect remaining ones
    remaining_tasks = []

    for task_properties in all_udf_tasks:
        # Check if this UDF targets this stage with the specified directive
        target_stage = task_properties.get("target_stage", "")
        run_before = task_properties.get("run_before", False)
        run_after = task_properties.get("run_after", False)

        # Determine if this UDF should execute
        should_execute = False
        if directive == "run_before" and run_before and target_stage == stage_name:
            should_execute = True
        elif directive == "run_after" and run_after and target_stage == stage_name:
            should_execute = True

        if should_execute:
            try:
                # Get UDF function details
                udf_function_str = task_properties.get("udf_function", "").strip()
                udf_function_name = task_properties.get("udf_function_name", "").strip()
                task_id = task_properties.get("task_id", "unknown")

                # Skip empty UDF functions
                if not udf_function_str:
                    logger.debug(f"UDF task {task_id} has empty function, skipping")
                    remaining_tasks.append(task_properties)
                    continue

                # Validate function name
                if not udf_function_name:
                    raise ValueError(f"UDF task {task_id} missing required 'udf_function_name' property")

                # Get or compile UDF function
                cached_udf = _udf_cache.get(udf_function_str, udf_function_name)
                if cached_udf:
                    udf_function = cached_udf.function
                    logger.debug(f"UDF task {task_id}: Using cached function '{udf_function_name}'")
                else:
                    udf_function = compile_and_validate_udf(udf_function_str, udf_function_name, task_id)
                    _udf_cache.put(udf_function_str, udf_function_name, udf_function)
                    logger.debug(f"UDF task {task_id}: Cached function '{udf_function_name}'")

                # Execute the UDF
                control_message = udf_function(control_message)

                # Validate return type
                if not isinstance(control_message, IngestControlMessage):
                    raise ValueError(f"UDF task {task_id} must return IngestControlMessage")

                logger.info(f"Executed UDF {task_id} '{udf_function_name}' {directive} stage '{stage_name}'")

            except Exception as e:
                logger.error(f"UDF {task_id} failed {directive} stage '{stage_name}': {e}")
                # Keep failed task for next stage
                remaining_tasks.append(task_properties)
        else:
            # Keep non-applicable task for next stage
            remaining_tasks.append(task_properties)

    # Re-add all remaining UDF tasks
    for task_properties in remaining_tasks:
        from nv_ingest_api.internal.primitives.control_message_task import ControlMessageTask

        task = ControlMessageTask(type="udf", id=task_properties.get("task_id", "unknown"), properties=task_properties)
        control_message.add_task(task)

    return control_message


def remove_task_by_id(control_message: IngestControlMessage, task_id: str) -> IngestControlMessage:
    """Remove a specific task by ID from the control message"""
    try:
        control_message.remove_task(task_id)
    except RuntimeError as e:
        logger.warning(f"Could not remove task {task_id}: {e}")

    return control_message


def udf_intercept_hook(stage_name: Optional[str] = None, enable_run_before: bool = True, enable_run_after: bool = True):
    """
    Decorator that executes UDFs targeted at this stage.

    This decorator integrates with the existing UDF system, providing full
    UDF compilation, caching, and execution capabilities. UDFs can target
    specific stages using run_before or run_after directives.

    Args:
        stage_name: Name of the stage (e.g., "image_dedup", "text_extract").
                   If None, will attempt to use self.stage_name from the decorated method's instance.
        enable_run_before: Whether to execute UDFs with run_before=True (default: True)
        enable_run_after: Whether to execute UDFs with run_after=True (default: True)

    Examples:
        # Automatic stage name detection (recommended)
        @traceable("image_deduplication")
        @udf_intercept_hook()  # Uses self.stage_name automatically
        @filter_by_task(required_tasks=["dedup"])
        def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
            return control_message

        # Explicit stage name (fallback/override)
        @traceable("data_sink")
        @udf_intercept_hook("data_sink", enable_run_after=False)
        @filter_by_task(required_tasks=["store"])
        def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
            return control_message

        # Only run_after UDFs (e.g., for source stages)
        @traceable("data_source")
        @udf_intercept_hook(enable_run_before=False)  # Uses self.stage_name automatically
        def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
            return control_message
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if UDF processing is globally disabled
            if os.getenv("INGEST_DISABLE_UDF_PROCESSING"):
                logger.debug("UDF processing is disabled via INGEST_DISABLE_UDF_PROCESSING environment variable")
                return func(*args, **kwargs)

            # Determine the stage name to use
            resolved_stage_name = stage_name

            # If no explicit stage_name provided, try to get it from self.stage_name
            if resolved_stage_name is None and len(args) >= 1:
                stage_instance = args[0]  # 'self' in method calls
                if hasattr(stage_instance, "stage_name") and stage_instance.stage_name:
                    resolved_stage_name = stage_instance.stage_name
                    logger.debug(f"Using auto-detected stage name: '{resolved_stage_name}'")
                else:
                    logger.warning(
                        "No stage_name provided and could not auto-detect from instance. Skipping UDF intercept."
                    )
                    return func(*args, **kwargs)
            elif resolved_stage_name is None:
                logger.warning(
                    "No stage_name provided and no instance available for auto-detection. Skipping UDF intercept."
                )
                return func(*args, **kwargs)

            # Extract control_message from args (handle both self.method and function cases)
            control_message = None
            if len(args) >= 2 and hasattr(args[1], "get_tasks"):
                control_message = args[1]  # self.method case
                args_list = list(args)
            elif len(args) >= 1 and hasattr(args[0], "get_tasks"):
                control_message = args[0]  # function case
                args_list = list(args)

            if control_message:
                # Execute UDFs that should run before this stage (if enabled)
                if enable_run_before:
                    control_message = execute_targeted_udfs(control_message, resolved_stage_name, "run_before")
                    # Update args with modified control_message
                    if len(args) >= 2 and hasattr(args[1], "get_tasks"):
                        args_list[1] = control_message
                    else:
                        args_list[0] = control_message

                # Execute the original stage logic
                result = func(*tuple(args_list), **kwargs)

                # Execute UDFs that should run after this stage (if enabled)
                if enable_run_after and hasattr(result, "get_tasks"):  # Result is control_message
                    result = execute_targeted_udfs(result, resolved_stage_name, "run_after")

                return result
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_udf_cache_stats() -> Dict[str, Any]:
    """Get UDF cache performance statistics"""
    return _udf_cache.get_stats()
