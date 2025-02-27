from morpheus.config import ExecutionMode
from morpheus.stages.general.linear_modules_source import LinearModuleSourceStage
from morpheus.stages.general.linear_modules_stage import LinearModulesStage


class LinearModuleSourceStageCPU(LinearModuleSourceStage):
    def supported_execution_modes(self) -> tuple[ExecutionMode]:
        # Provide your own logic here; for example:
        return (ExecutionMode.CPU,)


class LinearModuleStageCPU(LinearModulesStage):
    def supported_execution_modes(self) -> tuple[ExecutionMode]:
        # Provide your own logic here; for example:
        return (ExecutionMode.CPU,)
