import openseespy.opensees as osp

class LinearAnalysis:
    def run_analysis(self):
        osp.system("BandSPD")                   # Define the system of equations
        osp.numberer("RCM")                     # Define the numbering algorithm (Reverse Cuthill-McKee)
        osp.constraints("Plain")                # Define the constraint handler
        osp.integrator("LoadControl", 1.0)      # Define the integrator for the analysis (Load Control with step size 1.0)
        osp.algorithm("Linear")                 # Define the solution algorithm (Linear)
        osp.analysis("Static")                  # Define the type of analysis (Static)

        # Perform the analysis
        osp.analyze(1)  # Analyze one load step

__all__ = [
    "LinearAnalysis",
]