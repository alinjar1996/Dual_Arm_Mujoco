import numpy as np
import coacd
import trimesh
import xml.etree.ElementTree as ET
import argparse
import os
import yaml

parser = argparse.ArgumentParser(prog="ACD")
parser.add_argument("-o","--object",type=str,required=True)
parser.add_argument("-os","--ObjectScale",type=float,required=True)
parser.add_argument("-ms","--MujocoScale",type=float,default=1.0)

args = parser.parse_args()

name = args.object
scaleObj = args.ObjectScale
scaleVal = args.MujocoScale



# mesh_file = './grasps/objects/DA2-{}.obj'.format(name)
mesh_file = "../panda/meshes/visual/objects/tray.stl"

mesh = trimesh.load(mesh_file,force="mesh")
mesh.vertices -= mesh.centroid
mesh.apply_scale(scaleObj)

minz = -mesh.sample(512)[:,-1].min()

#! uncomment this later 
# with open("./config.yaml","r") as infile:
#     data = yaml.safe_load(infile)

#     data["parameters"]["ACD"]["minZ"] = float(minz)

#     with open("./config.yaml","w") as outfile:
#         yaml.dump(data,outfile)


#terminate script if acd file already exists
# if os.path.exists("./panda/assets/{}.xml".format(name)):

#     Xobj = ET.Element("mujocoinclude")
#     Xbody = ET.SubElement(Xobj,"body",attrib={"name":"collision_object","pos":"0.0 0.3 0.2","quat":"1 0 0 0"})
#     Xjoint = ET.SubElement(Xbody,"joint",attrib={"type":"free","name":"object_virtual_joint","pos":"0.0 0.0 0.0","damping":"5"})
#     XvisG = ET.SubElement(Xbody,"geom",attrib={"class":"object_viz","mesh":"{}_viz".format(name)})

#     numP = 0
#     for file in os.listdir("./panda/meshes/collision/objects/decompositions/"):
#         if "{}Part".format(name) in file:
#             numP +=1

#     for i in range(numP):
#         XcolG = ET.SubElement(Xbody,"geom",attrib={"class":"object_col","mesh":"{}Part-".format(name)+str(i)})

#     Xinert = ET.SubElement(Xbody,"inertial",attrib={"pos":"0 0 0","mass":"1","diaginertia":"2 2 2"})


#     XobTree = ET.ElementTree(Xobj)
#     with open("./object.xml", "wb") as f:
#         XobTree.write(f, encoding="utf-8")
    
#     Xinc = ET.Element("mujocoinclude")
#     Xfile = ET.SubElement(Xinc,"include",attrib={"file":"./panda/assets/{}.xml".format(name)})

#     XincTree = ET.ElementTree(Xinc)
#     with open("./include.xml", "wb") as f:
#         XincTree.write(f, encoding="utf-8")
#     exit(0)
#do mean center
# scaleObj = 0.024724145342293464

mesh.export("../panda/meshes/collision/objects/{}.stl".format(name))
mesh.export("../panda/meshes/visual/objects/{}.stl".format(name))


meshACD = coacd.Mesh(mesh.vertices,mesh.faces)
parts = coacd.run_coacd(meshACD, resolution=2000, mcts_iterations=100, mcts_max_depth=5)
scale = "{} {} {}".format(scaleVal,scaleVal,scaleVal)

numParts = 0

Xmjinc = ET.Element("mujocoinclude")
Xasset = ET.SubElement(Xmjinc,"asset")
XcolM = ET.SubElement(Xasset,"mesh",attrib={"name":"{}_viz".format(name),"file":"../meshes/visual/objects/{}.stl".format(name),"scale":scale})

for i,part in enumerate(parts):
    Pmesh = trimesh.Trimesh(part[0],part[1])
    path = "../panda/meshes/collision/objects/decompositions/tray/"
    filename = name+"Part-"+str(i)+".stl"
    Pmesh.export(path+filename)
    Xatrdict = {"name":filename.rsplit(".",1)[0],"file":"../meshes/collision/objects/decompositions/tray/"+filename, "scale":scale}
    Xmesh = ET.SubElement(Xasset,"mesh",attrib=Xatrdict)
    numParts += 1

XAssetTree = ET.ElementTree(Xmjinc)

with open("../panda/assets/{}.xml".format(name), "wb") as f:
    XAssetTree.write(f, encoding="utf-8")

print("asset XML file created successfully.")

Xobj = ET.Element("mujocoinclude")
Xbody = ET.SubElement(Xobj,"body",attrib={"name":"collision_object","pos":"0.0 0.3 0.2","quat":"1 0 0 0"})
Xjoint = ET.SubElement(Xbody,"joint",attrib={"type":"free","name":"object_virtual_joint","pos":"0.0 0.0 0.0","damping":"5"})
XvisG = ET.SubElement(Xbody,"geom",attrib={"class":"object_viz","mesh":"{}_viz".format(name)})

for i in range(numParts):
    XcolG = ET.SubElement(Xbody,"geom",attrib={"class":"object_col","mesh":"{}Part-".format(name)+str(i)})

Xinert = ET.SubElement(Xbody,"inertial",attrib={"pos":"0 0 0","mass":"5","diaginertia":"2 2 2"})


XobTree = ET.ElementTree(Xobj)
with open("./object.xml", "wb") as f:
    XobTree.write(f, encoding="utf-8")

print("body XML file created successfully.")

Xinc = ET.Element("mujocoinclude")
Xfile = ET.SubElement(Xinc,"include",attrib={"file":"./panda/assets/{}.xml".format(name)})

XincTree = ET.ElementTree(Xinc)
with open("./include.xml", "wb") as f:
    XincTree.write(f, encoding="utf-8")