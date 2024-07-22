from analysis import amber_minimize
from typing import Sequence, Dict, Any, Tuple
import numpy as np

_RELAX_STIFFNESS = 10.0

class AmberRelaxation(object):
  """Amber relaxation."""

  def __init__(self,
               *,
               max_iterations=2,
               stiffness=_RELAX_STIFFNESS,
               tolerance=1e-2,
               max_outer_iterations=3,
               use_gpu=False):
    """Initialize Amber Relaxer.

    Args:
      max_iterations: Maximum number of L-BFGS iterations. 0 means no max.
      tolerance: kcal/mol, the energy tolerance of L-BFGS.
      stiffness: kcal/mol A**2, spring constant of heavy atom restraining
        potential.
      exclude_residues: Residues to exclude from per-atom restraining.
        Zero-indexed.
      max_outer_iterations: Maximum number of violation-informed relax
       iterations. A value of 1 will run the non-iterative procedure used in
       CASP14. Use 20 so that >95% of the bad cases are relaxed. Relax finishes
       as soon as there are no violations, hence in most cases this causes no
       slowdown. In the worst case we do 20 outer iterations.
      use_gpu: Whether to run on GPU.
    """

    self._max_iterations = max_iterations
    self._tolerance = tolerance
    self._max_outer_iterations = max_outer_iterations
    self._use_gpu = use_gpu
    self.stiffness = stiffness

  def process(self, 
              pdb_string,
              ) -> Tuple[str, Dict[str, Any], Sequence[float]]:
    """Runs Amber relax on a prediction, adds hydrogens, returns PDB string."""
    out = amber_minimize.run_pipeline(
        pdb_string=pdb_string, max_iterations=self._max_iterations,
        stiffness=self.stiffness,
        tolerance=self._tolerance, 
        max_outer_iterations=self._max_outer_iterations,
        use_gpu=self._use_gpu)
    min_pos = out['pos']
    start_pos = out['posinit']
    rmsd = np.sqrt(np.sum((start_pos - min_pos)**2) / start_pos.shape[0])
    debug_data = {
        'initial_energy': out['einit'],
        'final_energy': out['efinal'],
        'attempts': out['min_attempts'],
        'rmsd': rmsd
    }
    min_pdb = out['min_pdb']
    return min_pdb, debug_data