# Estimating Force Interactions of Deformable Linear Objects from their Shapes
Using the jpQ-DER model based off the DER theory ([Bergou2008](http://www.cs.columbia.edu/cg/pdfs/143-rods.pdf)), the external force interactions of a deformable linear object (DLO) can be estimated using a set of consistency equations and solving a set of linear equations based on the force-torque balancce of the system.

## Requires
Requires: [MuJoCo](https://mujoco.readthedocs.io/en/latest/overview.html), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), [mujoco-python-viewer](https://github.com/rohanpsingh/mujoco-python-viewer), [eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download)

## Use:
1. In the root directory of this package:
```
pip install -e .
```
2. Put all data required for real experiments in the same directory level as the repository:
```
cd ..
mkdir data && cd data
mkdir datadir1/outputs/pos3d
```
Put your .npy posfiles in /pos3d directory.
3. To run interactive simulation:
```
python interactive_sim.py
```
then use the mouse to interact with the wire. Double-click to select a body, right-click pull to apply force.
4. To run real experiments:
```
python wirestandalone_example.py --data_dir datadir1 --pos_file posfilename.npy
```
5. To plot results for real experiments:
```
python plot_s2fresults.py --data_dir datadir1 --pos_file posfilename.npy
```