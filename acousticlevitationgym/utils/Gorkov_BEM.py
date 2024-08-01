from acoustools.Mesh import load_scatterer
from acoustools.Utilities import TOP_BOARD, create_points
from acoustools.Solvers import wgs
from acoustools.BEM import compute_E, propagate_BEM_pressure, BEM_gorkov_analytical, propagate_BEM, get_cache_or_compute_H
import acoustools.Constants as c
import torch
 

path = r"/home/william/BEMMedia/flat-lam2.stl"
 
reflector = load_scatterer(path,dz=-0.06)
 
p = create_points(N=4)
print(p.shape)
 
H = get_cache_or_compute_H(reflector, TOP_BOARD, path=r"/home/william/BEMMedia")
 
E = compute_E(reflector,points=p,board=TOP_BOARD,path=r"/home/william/BEMMedia",H=H)
x = wgs(p,iter=5,A=E)
  
trap_up = p
trap_up[:,2] += c.wavelength/4
print(trap_up)
print(propagate_BEM_pressure(x,trap_up,reflector,TOP_BOARD,path=r"/home/william/BEMMedia",H=H))
print(BEM_gorkov_analytical(x,trap_up,reflector,TOP_BOARD,path=r"/home/william/BEMMedia",H=H))