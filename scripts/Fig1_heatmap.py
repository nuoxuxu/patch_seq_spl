import patch_seq_spl.helper_functions as src
from importlib import reload
reload(src)

src.plot_glm_results("proc/three/simple", 100, save_path = "results/figures/Fig1_three_simple.png")
src.plot_glm_results("proc/three/multiple", 40, vmax = 20, save_path = "results/figures/Fig1_three_multiple.png")
src.plot_glm_results("proc/five/simple", 100, save_path = "results/figures/Fig1_five_simple.png")
src.plot_glm_results("proc/five/multiple", 40, vmax = 20, save_path = "results/figures/Fig1_five_multiple.png")