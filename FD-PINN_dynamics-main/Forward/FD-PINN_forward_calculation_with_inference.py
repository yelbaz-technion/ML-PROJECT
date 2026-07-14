import os  # Provides file paths, directory creation, and environment variables.
os.environ["DDE_BACKEND"] = "pytorch"  # Tell DeepXDE to use PyTorch before importing it.

import deepxde as dde  # PINN geometry, derivatives, boundary conditions, networks, and training.
import numpy as np  # Numerical arrays, grids, reshaping, and text-file output.
import math  # Mathematical constants; used here for pi.
import shutil  # File utilities (currently unused).
import torch  # Differentiable tensor operations used by the PDE and transforms.
from deepxde import utils  # DeepXDE utilities (currently unused).
from deepxde.icbc.boundary_conditions import npfunc_range_autocache  # Currently unused.
import time  # Timing utilities (currently unused).
import matplotlib  # Base plotting package (currently unused directly).
import matplotlib.pyplot as plt  # Creates and saves plots.
import pickle  # Object serialization (currently unused).
import scipy.io  # MATLAB file I/O (currently unused).

script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory containing this script.

PI = math.pi  # Mathematical constant pi.
m=1  # Beam mass per unit length.
ks=0.025  # Damping-related coefficient coupling real and imaginary responses.
EI=500  # Beam flexural rigidity (Young's modulus times area moment of inertia).
v=1  # Moving-load speed.
L=10  # Beam length; position x ranges from 0 to L.
p0=20  # Applied-load magnitude.
w=(PI**2/L**2)*(EI/m)**0.5  # First natural angular frequency (not used later).
fmax=0.5  # Maximum frequency in the PINN input domain.

dde.config.real.set_float64()  # Use 64-bit real values inside DeepXDE.
dde.config.set_default_float("float64")  # Improve precision of fourth-order derivatives.

geom = dde.geometry.Interval(0, L)  # Define the spatial interval 0 <= x <= L.
timedomain = dde.geometry.TimeDomain(0, fmax)  # Used as frequency f, despite its name.
geomtime = dde.geometry.GeometryXTime(geom, timedomain)  # Combined (x, f) domain.

xx = np.linspace(0, L, 21)    # Create 21 uniformly spaced positions.
ww = np.linspace(0, fmax, 21)  # Create 21 uniformly spaced frequencies.
xx, ww = np.meshgrid(xx, ww)  # Form all 21 x 21 position-frequency pairs.

xx = xx.flatten()[:, None]  # Convert the position grid to a (441, 1) column.
ww = ww.flatten()[:, None]  # Convert the frequency grid to a (441, 1) column.

AN = np.concatenate((xx, ww), 1)  # Create 441 fixed collocation points [x, f].

def pde(x, y):  # Define the two governing-equation residuals minimized during training.

    ur_xx=dde.grad.hessian(y, x,component=0, i=0, j=0)  # Calculate d²ur/dx².
    ur_xxxx=dde.grad.hessian(ur_xx, x, i=0, j=0)  # Calculate d⁴ur/dx⁴.
    ui_xx=dde.grad.hessian(y, x,component=1, i=0, j=0)  # Calculate d²ui/dx².
    ui_xxxx=dde.grad.hessian(ui_xx, x, i=0, j=0)  # Calculate d⁴ui/dx⁴.

    # Real residual: bending + damping coupling + inertia + cosine load.
    eq_ur=EI*ur_xxxx-2*ks*y[:,1:2]*(x[:, 1:2]*2*PI)**2-m*y[:,0:1]*(x[:, 1:2]*2*PI)**2-p0/v*(torch.cos(x[:, 0:1]*x[:, 1:2]*2*PI/v))
    # Imaginary residual: bending + damping coupling + inertia + sine load.
    eq_ui=EI*ui_xxxx+2*ks*y[:,0:1]*(x[:, 1:2]*2*PI)**2-m*y[:,1:2]*(x[:, 1:2]*2*PI)**2+p0/v*(torch.sin(x[:, 0:1]*x[:, 1:2]*2*PI/v))

    return [eq_ur,eq_ui]  # Return two residuals, which become the first two loss terms.

def boundary_l(x, on_boundary):  # Select points on the beam's spatial boundary.

    return on_boundary  # True at both x=0 and x=L, despite the function's name.


def fun_ur_xx(x, y, _):  # Operator for the real bending-moment boundary condition.
    ur_xx=dde.grad.hessian(y, x,component=0, i=0, j=0)  # Calculate d²ur/dx².
    return ur_xx  # OperatorBC will force this value to zero at both ends.

def fun_ui_xx(x, y, _):  # Operator for the imaginary bending-moment condition.
    ui_xx=dde.grad.hessian(y, x,component=1, i=0, j=0)  # Calculate d²ui/dx².
    return ui_xx  # OperatorBC will force this value to zero at both ends.


bc1 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_l,component=0)  # ur=0 at both ends.
bc2 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_l,component=1)  # ui=0 at both ends.
bc3 = dde.icbc.OperatorBC(geomtime, fun_ur_xx, boundary_l)  # d²ur/dx²=0 at both ends.
bc4 = dde.icbc.OperatorBC(geomtime, fun_ui_xx, boundary_l)  # d²ui/dx²=0 at both ends.


data = dde.data.TimePDE(  # Package the physics problem for DeepXDE.
    geomtime,  # Use the combined position-frequency domain.
    pde,  # Use the residual function defined above.
    [bc1,bc2,bc3,bc4],  # Include the four simply supported boundary conditions.
    num_domain=0,  # Generate no additional random interior points.
    num_boundary=0,  # Generate no additional random boundary points.
    num_initial=0,  # Generate no initial-time points; the second coordinate is frequency.
    anchors=AN  # Use the fixed 21 x 21 grid as collocation points.
)

layer_size = [2] + [60] * 3 + [2]  # Architecture [2, 60, 60, 60, 2].
activation = "tanh"  # Smooth activation suitable for fourth derivatives.
initializer = "Glorot uniform"  # Xavier/Glorot uniform weight initialization.
net = dde.nn.FNN(  # Build the fully connected network mapping (x,f) to (ur,ui).
    layer_size, activation, initializer  # Pass its architecture, activation, and initializer.
)

def feature_transform(x):  # Normalize the two inputs before they enter the network.

    x0=x[:,0:1]/L  # Scale position from [0,L] to [0,1].
    x1=x[:,1:2]/fmax  # Scale frequency from [0,fmax] to [0,1].

    return torch.cat(  # Recombine both normalized columns into an (N,2) tensor.
        [x0,x1], dim=1  # Concatenate along the feature/column dimension.
    )
def output_transform(x, y):  # Scale the network's raw output values.

    return torch.cat([y[:,0:1]*10,y[:,1:2]*10], dim=1)  # Multiply ur and ui by 10.



net.apply_feature_transform(feature_transform)  # Attach input normalization to the network.

net.apply_output_transform(output_transform)  # Attach output scaling to the network.

model = dde.Model(data, net)  # Combine the problem definition and network into a PINN.
maxiter = 2 
dde.optimizers.set_LBFGS_options(ftol=0,gtol=1e-20,maxiter=maxiter)  # Configure  L-BFGS iterations.
optimizer_name = "L-BFGS"  # Text displayed later in the loss-plot title.
# Compile with L-BFGS; the weights correspond to 2 PDE + 4 boundary-condition losses.
model.compile("L-BFGS", loss_weights=[1,1,1e2,1e2,1e6,1e6])

model_dir = os.path.join(script_dir, "model")  # Path of the checkpoint directory.
os.makedirs(model_dir, exist_ok=True)  # Create that directory if it does not exist.
checkpoint_base = os.path.join(model_dir, "trained-model")  # Checkpoint filename prefix.
checkpoint_path = os.path.join(model_dir, "trained-model-20000.pt")  # Exact forward checkpoint path.
#model.restore(checkpoint_path, verbose=1)  # Restore matching network and L-BFGS optimizer states.

# Override DeepXDE's stale 1000-iteration PyTorch block.
model.opt.param_groups[0]["max_iter"] = maxiter
model.opt.param_groups[0]["max_eval"] = 3


pde_residual_resampler = dde.callbacks.PDEPointResampler(period=100)  # Resample every 100 steps.


class IterationPrinter(dde.callbacks.Callback):  # Custom progress-printing callback.
    def on_epoch_end(self):  # DeepXDE calls this at the end of every epoch.
        step = self.model.train_state.step  # Read the current training step.
        if step % 100 == 0:  # Print only at multiples of 100.
            print(f"Iteration {step}")  # Display the iteration number.

iteration_printer = IterationPrinter()  # Instantiate the custom callback.
# Continue L-BFGS optimization, record losses, run callbacks, and save checkpoints.

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Model device:", next(model.net.parameters()).device)

losshistory, train_state = model.train(display_every=1,callbacks=[pde_residual_resampler, iteration_printer],model_save_path=checkpoint_base)

loss_test = np.sum(losshistory.loss_test, axis=1)  # Sum all six test losses at each step.
np.savetxt(os.path.join(script_dir, 'loss_test.txt'), loss_test)  # Save total test loss.

loss_train = np.sum(losshistory.loss_train, axis=1)  # Sum all six training losses.
np.savetxt(os.path.join(script_dir, 'loss_train.txt'), loss_train)  # Save total train loss.

dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir=script_dir)  # Save DeepXDE data.

no=501  # Original sample-count variable; currently unused.
x_star = np.linspace(0, L, 101)  # Create 101 positions for prediction.
f_star = np.linspace(0, fmax, 41)  # Create 41 frequencies for prediction.

X1_star, F1_star = np.meshgrid(x_star,f_star)  # Form all 101 x 41 evaluation pairs.
F1_star = F1_star.flatten()[:, None]  # Flatten frequencies to a (4141,1) column.
X1_star = X1_star.flatten()[:, None]  # Flatten positions to a (4141,1) column.


X = np.vstack((np.ravel(X1_star), np.ravel(F1_star))).T  # Build (4141,2) [x,f] inputs.

pred = model.predict(X)  # Evaluate the trained PINN at all prediction points.

ur_pred = pred[:, 0]  # Extract the predicted real response.
ui_pred = pred[:, 1]  # Extract the predicted imaginary response.

# Reshape the flat predictions into grids whose rows are positions and columns are frequencies.
ur_grid = ur_pred.reshape(len(f_star), len(x_star)).T
ui_grid = ui_pred.reshape(len(f_star), len(x_star)).T

np.savetxt(os.path.join(script_dir, 'X1_star.txt'), X1_star)  # Save prediction positions.
np.savetxt(os.path.join(script_dir, 'F1_star.txt'), F1_star)  # Save prediction frequencies.

np.savetxt(os.path.join(script_dir, 'ur_pred.txt'), ur_pred)  # Save real predictions.
np.savetxt(os.path.join(script_dir, 'ui_pred.txt'), ui_pred)  # Save imaginary predictions.

plt.figure()  # Create the loss-history figure.
plt.semilogy(losshistory.steps, loss_train, label="Train loss")  # Plot train loss on log scale.
plt.semilogy(losshistory.steps, loss_test, label="Test loss")  # Plot test loss on log scale.
plt.xlabel("# Steps")  # Label the horizontal axis.
plt.ylabel("Loss")  # Label the vertical axis.
plt.title(f"Forward PINN Loss History ({optimizer_name})")  # Add the plot title.
plt.legend()  # Show the train/test legend.
plt.tight_layout()  # Prevent labels from being clipped.
plt.savefig(os.path.join(script_dir, "forward_loss_history.png"), dpi=300)  # Save the PNG.

fig_ur, ax_ur = plt.subplots()  # Create axes for the real-response heatmap.
heatmap_ur = ax_ur.pcolormesh(  # Draw frequency horizontally and position vertically.
    f_star, x_star, ur_grid, shading="auto", cmap="RdBu_r"
)
ax_ur.set_xlabel("Frequency f")  # Label the horizontal frequency axis.
ax_ur.set_ylabel("Position x")  # Label the vertical position axis.
ax_ur.set_title("Forward PINN Real Response $u_r(x, f)$")  # Add the plot title.
fig_ur.colorbar(heatmap_ur, ax=ax_ur, label="$u_r$")  # Map colors to real displacement.
fig_ur.tight_layout()  # Prevent labels from being clipped.
fig_ur.savefig(os.path.join(script_dir, "forward_ur_heatmap.png"), dpi=300)  # Save the PNG.

fig_ui, ax_ui = plt.subplots()  # Create axes for the imaginary-response heatmap.
heatmap_ui = ax_ui.pcolormesh(  # Draw frequency horizontally and position vertically.
    f_star, x_star, ui_grid, shading="auto", cmap="RdBu_r"
)
ax_ui.set_xlabel("Frequency f")  # Label the horizontal frequency axis.
ax_ui.set_ylabel("Position x")  # Label the vertical position axis.
ax_ui.set_title("Forward PINN Imaginary Response $u_i(x, f)$")  # Add the plot title.
fig_ui.colorbar(heatmap_ui, ax=ax_ui, label="$u_i$")  # Map colors to imaginary displacement.
fig_ui.tight_layout()  # Prevent labels from being clipped.
fig_ui.savefig(os.path.join(script_dir, "forward_ui_heatmap.png"), dpi=300)  # Save the PNG.

plt.show()  # Display all figures interactively.
