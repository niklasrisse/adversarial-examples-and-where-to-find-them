using MIPVerify
using Gurobi
using JuMP
using Images
using Printf
using MAT
using CSV
using NPZ

# import model
mat_mip_file = "../../provable_robustness_max_linear_regions/models/mmr+at/2019-02-14 23:20:04 dataset=mnist nn_type=fc1 p_norm=inf lmbd=1.0 gamma_rb=0.1 gamma_db=0.1 stage1hpl=10 ae_frac=0.5 epoch=100_mip.mat"
param_dict = mat_mip_file |> matread;

# import dataset
dataset = MIPVerify.read_datasets("MNIST")

# build model 
n_h = size(dataset.test.images, 2)
n_w = size(dataset.test.images, 3)
n_c = size(dataset.test.images, 4)
n_out = length(unique(dataset.test.labels))

n_in = n_h * n_w * n_c

n_hidden = 1024
fc1 = get_matrix_params(param_dict, "fc1", (n_in, n_hidden))
softmax = get_matrix_params(param_dict, "softmax", (n_hidden, n_out))

n1 = Sequential([
    Flatten(4),
    fc1, ReLU(interval_arithmetic),
    softmax
    ],
    "outputs/"
)

for i = 1:100

    sample_image = MIPVerify.get_image(dataset.test.images, i);
    sample_label = MIPVerify.get_label(dataset.test.labels, i) + 1;

    d = MIPVerify.find_adversarial_example(n1, 
    sample_image, 
    sample_label,
    invert_target_selection=true,
    GurobiSolver(OutputFlag=0), 
    rebuild=true,
    cache_model=false,
    tightening_algorithm=lp, 
    tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0),
    norm_order=Inf);

    perturbed_sample_image = getvalue(d[:PerturbedInput])

    npzwrite("outputs/inputs_for_exact_adversarial_$(@sprintf("%.0f", i)).npz", sample_image)
    npzwrite("outputs/exact_adversarial_$(@sprintf("%.0f", i)).npz", perturbed_sample_image)

end