import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt

# Clear existing model
ops.wipe()

# Create 2D model
ops.model('basic', '-ndm', 2, '-ndf', 2)

# Define geometry
# Nodes: (tag, x, y)
ops.node(1, 0.0, 0.0)
ops.node(2, 4.0, 0.0)
ops.node(3, 2.0, 3.0)

# Define boundary conditions
ops.fix(1, 1, 1)  # Fix node 1 in x and y directions
ops.fix(2, 1, 1)  # Fix node 2 in x and y directions

# Define materials and elements
E = 200e9  # Young's modulus (Pa)
A = 0.01   # Cross-sectional area (m²)

# Truss material
ops.uniaxialMaterial('Elastic', 1, E)

# Truss elements
ops.element('Truss', 1, 1, 3, A, 1)
ops.element('Truss', 2, 2, 3, A, 1)

# Visualize the truss (same as in Method 2)
def plot_truss():
    plt.figure(figsize=(8, 6))
    # Plot nodes
    for i in range(1, 4):
        x, y = ops.nodeCoord(i)
        plt.plot(x, y, 'bo', markersize=10)
        plt.text(x+0.1, y+0.1, f'Node {i}')

    # Plot elements
    plt.plot([0, 2], [0, 3], 'k-', linewidth=2)
    plt.text(1, 1.5, 'Element 1')
    plt.plot([4, 2], [0, 3], 'k-', linewidth=2)
    plt.text(3, 1.5, 'Element 2')

    plt.grid(True)
    plt.axis('equal')
    plt.title('Simple Truss Model')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()

#plot_truss()

# Method 3: Using system commands
# Set up analysis for matrix extraction
print("start")
ops.wipeAnalysis()
print('wipe')

ops.system('FullGeneral')  # Use FullGeneral to be able to extract the matrix
print('sys')
ops.numberer('Plain')  # Use Plain numberer to get 0-based indexing
print('num')
ops.constraints('Plain')
print('const')
ops.integrator('LoadControl', 1.0, 1, 1, 1)
print('lc')
ops.algorithm('Linear')
print('alg')
ops.analysis('Static')
print('anal')
ops.analyze(0)
print('done')
"""
# Get the stiffness matrix
try:
    # In newer OpenSeesPy versions
    K_full = ops.printA('-ret')
    # Since only node 3 is free (DOFs 4 and 5 in 0-based indexing),
    # extract the relevant 2x2 submatrix
    K = K_full[4:6, 4:6]  # This is for 0-based indexing
except:
    print("printA command not supported. You may need a newer OpenSeesPy version.")
    K = np.zeros((2, 2))  # Placeholder

print("Stiffness Matrix (Method 3):")
"""
print(ops.printA('-file', './a.out'))