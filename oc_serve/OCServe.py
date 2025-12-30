"""Main OCServe Application"""
from configs import OCServeConfigs
from .orchestrators import Orchestrator

class OCServe():
    """Main OCServe Application"""
    def __init__(self):
        self.configs = OCServeConfigs()
        self.deployment = Orchestrator.get(self.configs.orchestrator_type)

    def get_deployment(self):
        """Returns the current deployment orchestrator"""
        return self.deployment
