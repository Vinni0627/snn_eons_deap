BASE_RES="./results_ablation"
BASE_FIG="./figures_ablation"

INT=("int_1" "int_3" "int_5")
OBJ=("obj_15" "obj_45" "obj_90")


for i in "${!INT[@]}"; do
    int="${INT[$i]}"
    obj="${OBJ[$i]}"

    int_res="${BASE_RES}/${int}"
    int_fig="${BASE_FIG}/${int}"
    obj_res="${BASE_RES}/${obj}"
    obj_fig="${BASE_FIG}/${obj}"

    python3 plot_hybrid_results.py --res_dir "$int_res" --fig_dir "$int_fig"
    python3 plot_hybrid_results.py --res_dir "$obj_res" --fig_dir "$obj_fig"
    python3 plot_topology.py --res_dir "$int_res" --fig_dir "$int_fig"
    python3 plot_topology.py --res_dir "$obj_res" --fig_dir "$obj_fig"
done