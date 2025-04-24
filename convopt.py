import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse as ap
from scipy.spatial.transform import Rotation as R
import logging 
import pickle


from controllers.convex import Convex
from controllers.utils.QuinticPolynomial import *
from controllers.utils.utils import *
from controllers.planner import Planner
from controllers.utils.solvers import QuinticSolver
from icecream import ic
import random
import copy
import os

np.random.seed(56)
random.seed(56)

# model_path = "./models_dual/dual_panda_monitor_simple.xml"
# model_path = "./models_dual/dual_panda_cart.xml"
model_path = "./models_dual/dual_panda_tray.xml"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger("CONTROLLER")

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(
    model=model,
    data=data,
    show_left_ui=False,
    show_right_ui=False)


KvPos = 1
KvOri = 1

dt = 0.01 # 0.002
model.opt.timestep = dt



def main():
    # K = np.array([100,100,100,100,100,100]) * 1
    K = np.array([10,10,10,10, 10,10])*50
    K_null = np.array([100.0, 100.0, 55.0, 55.0, 22.5, 22.5, 5.0, 2.0, 2.0,
                        100.0, 100.0, 55.0, 55.0, 22.5, 22.5, 5.0, 2.0, 2.0]) 
    
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    controller = Convex(model, data, viewer)

    Util = RotationUtils()
    # model.opt.gravity[2] = -12.81
    controller.SetStaticParams(K,K_null,KvPos,KvOri,dt,final_object_pose= ([-0.2,0.0,0.4], [1,0, 0, 0]))
    # mass_normal = np.random.normal(loc=2750, scale=750)
    # mass = np.clip(mass_normal, 500, 5000)/1000
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "collision_object")
    # model.body_mass[body_id] = mass
    # mujoco.mj_setMassFromGeom(model, body_id)
    controller.resetViewer(True)

    planner = Planner(model,data,controller,QuinticSolver())
    
    # postBias = data.qpos[controller.dof_ids[:18]]
    velBias = data.qvel[controller.dof_ids[:18]]

    postBias = np.random.randn(1,18)[0].tolist()

    jacPL = controller.JL
    jacPR = controller.JR

    Wimp = 1
    Wpos = 0.01

    Qrange = np.array([-3.07,3.75])
    # Qdotrange = np.array([-2.6,2.6])
    # Qrange = np.array([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,3,3],[2.8973,1.7628, 2.8973, 0.0698, 2.8973, 3.7525, 2.8973,3,3]])
    Qdotrange = np.array([[-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100,-2.6100,-2.6100],[2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,2.6100,2.6100]])
    tauRange = np.array([[-87,-87,-87,-87,-12,-12,-12,-12,-12],[87,87,87,87,12,12,12,12,12]])
    # tauRange = np.array([[-500] * 9, [500] * 9])
    

    graspIdx = 0
    name = "chair-new"

    # grasps = np.load("/home/faizal/Documents/saad/DiffOpt-RL/chair_good_grasps.npy".format(name))
    grasps = np.load('/home/faizal/Documents/saad/DiffOpt-RL/examples/generatedGrasps/grasps_pot.npy')
    # print(grasps[graspIdx])
    
    grasps_tray =  np.array([[[[0.000796326711, -0.000796326458, -0.999999366, -0.0],
   [0.999999683, 6.3413623e-7, 0.000796326458, -0.25],
   [0, -0.999999683, 0.000796326711, 0.16+0.3],
   [0, 0, 0, 1]],

  [[0.000796326711, -0.000796326458, -0.999999366, -0.0],
   [0.999999683, 6.3413623e-7, 0.000796326458, 0.25],
   [0, -0.999999683, 0.000796326711, 0.16+0.3],
   [0, 0, 0, 1]]],


 [[[2.220446049250313e-16, 0, -1, 0],
   [0, 1, 0, -0.325],
   [1, 0, 2.220446049250313e-16, 0.233],
   [0, 0, 0, 1]],

  [[2.220446049250313e-16, 1.2246467991473532e-16, -1, 0],
   [0, -1, -1.2246467991473532e-16, 0.325],
   [-1, 2.465190328815662e-32, -2.220446049250313e-16, 0.233],
   [0, 0, 0, 1]]]])
        
    print(grasps.shape)
    ic(model.body_mass[body_id])

    graspL = grasps_tray[graspIdx][0]
    graspR = grasps_tray[graspIdx][1]
    
    object_scale = 1
    # objStrPos = [-0.40, 0.0, -0.07] #! round table z should be 0
    objStrPos = [-0.40, 0.0, -0.07] #! round table z should be 0
    
    objStrOri = [0,0,0]

    # final_object_pose = ([-0.5, -0.1, 0.422], [1, 0, 0, 0])
    final_object_pose = ([-0.5, 0, 0.5], [0.7071068,0.7071068, 0, 0])

    numsteps = 300

    logger.info("INITIALISING TASK")
    planner.setPlan([graspL,graspR],numsteps,object_scale,[objStrPos,objStrOri])

    #initial trajectories
    trajL,trajR = planner.makePlan(final_object_pose,PreGrasp=True,ClosedChain=False)
    i = 0
    j = 0
    k = 0
    l = 0
    stage = 1
    loss = []
    tausL = []
    tausR = []
    ope = []
    object_pos_err = []
    FDotR = []
    FDotL = []
    FR = []
    FL = []
    lossMargin = 1000000000000
    eucl_dist = []
    force_eef_L = []
    force_eef_R = []
    gripper_close_amt = 0.04
    
    
    logger.info("STARTING APPROACH TO OBJECT")
    while viewer.is_running():
        # force_eef_L = copy.deepcopy(data.sensor("frceef0").data)
        # force_eef_R = copy.deepcopy(data.sensor("frceef1").data)
        # slope_R = (force_eef_R - force_eef_R_Prev) / dt
        # slope_L = (force_eef_L - force_eef_L_Prev) / dt

        # # ic(slope_L,slope_R)
        # force_eef_L_Prev = copy.deepcopy(force_eef_L)
        # force_eef_R_Prev = copy.deepcopy(force_eef_R)
        
        if(i <numsteps and stage == 1):
            planner.update(trajL,trajR,i)
            i+=1
            
        force_increment_timesteps = np.concatenate([np.linspace(0, 1, 25), np.linspace(1, 0, 25)])
        
        # if stage == 3 and (1000 > k >= 500):
        #     body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "collision_object")
        #     # F = 150 - (15 * (k - 200))
        #     # force = -90 * force_increment_timesteps[k - 500]
        #     data.xfrc_applied[body_id] = [0, 0, -80, 0, 0, 0]
        #     print("APPLYING FORCE", k)
        #     ic(model.body_mass[body_id], force_eef_L, force_eef_R)
        #     controller.K = np.diag(np.array([100] * 6))
        # elif k < 500:
        #     controller.K = np.diag(np.array([300.0] * 6))
        
        # else:
        #     controller.K = np.diag(np.array([100.0] * 6))
            

        if(i==numsteps and stage==1):
            # ic(data.qpos[controller.dof_ids[:18]])
            logger.info("WAITING FOR ERROR RESOLUTION: ERROR MARGIN : 2000 || LOSS = {}".format(lossT))
            if(lossT<lossMargin):
                stage+=1
                logger.info("STAGE CHANGE")
                numsteps = 250
                planner.numsteps = numsteps
                trajL,trajR = planner.makePlan(final_object_pose,PreGrasp=False,ClosedChain=False)

        if(j<numsteps and stage==2):
            # ic(data.qpos[controller.dof_ids[:18]])
            planner.update(trajL,trajR,j)
            j+=1

        if(j==numsteps and stage ==2):
            logger.info("WAITING FOR ERROR RESOLUTION: ERROR MARGIN : 2000 || LOSS = {}".format(lossT))

            if(lossT<lossMargin):
                logger.info("STAGE CHANGE")
                stage+=1
                numsteps = 100
                planner.numsteps = numsteps
                trajL,trajR = planner.makePlan(final_object_pose,PreGrasp=False,ClosedChain=True)
                obj_traj = copy.deepcopy(planner.object_trajectory)
        
        if(k<numsteps and stage ==3):
            # print("K updated")
            current_obj_pose = data.body("collision_object").xpos.copy() 
            eucl_dist.append(np.linalg.norm(np.array(current_obj_pose) - np.array(obj_traj[k][0])))
            controller.K = np.diag(np.array([1, 1, 1, 1,1, 1]) * 500)
                                       
            # if True:
            #     if (500 >= k >= 300):
            #         # controller.K = np.diag(np.array([100, 100, min(100 + k*50 , 500) , 100, 100, 100] ))
            #         controller.K = np.diag(np.array([50, 50, 50, 50, 50, 50])) * 2
            #     else:
            #         controller.K = np.diag(np.array([100] * 6))
            # print(controller.K)
            # planner.update(trajL,trajR,k)
            k+=1
            
        if k == numsteps and stage == 3:
            logger.info("WAITING FOR ERROR RESOLUTION: ERROR MARGIN : 2000 || LOSS = {}".format(lossT))

            if(lossT<lossMargin):
                logger.info("STAGE CHANGE")
                stage+=1
                numsteps = 400
                planner.numsteps = numsteps
                # final_object_pose = ([-0.5, 0.25, 0.4], [0.7071068,0.7071068, 0, 0])
                final_object_pose = ([-0.5, 0.25, 0.4], [1,0,0,0])
                trajL,trajR = planner.makePlan(final_object_pose,PreGrasp=False,ClosedChain=True)
                obj_traj = copy.deepcopy(planner.object_trajectory)
                # print("hiiiiiiiiiiiiiiiiiii")
        if(l<numsteps and stage==4):
            # ic(data.qpos[controller.dof_ids[:18]])
            planner.update(trajL,trajR,l)
            # print("HELLO")
            l+=1
        if l == numsteps and stage == 4:
            # print("DK WHY")
            break
        # if(k==numsteps and stage==3):
        #     logger.info(f"WAITING FOR ERROR RESOLUTION: ERROR MARGIN : 2000 || LOSS = {lossT} || stage={stage}")
        #     current_obj_pose = data.body("collision_object").xpos.copy() 
        #     eucl_dist.append(np.linalg.norm(np.array(current_obj_pose) - np.array(obj_traj[numsteps - 1][0])))
        #     break
        #     if(lossT<lossMargin):
        #         stage+=1
        #         logger.info("STAGE CHANGE")
                
                
        


        # tauL, tauR,lossT, dist_obj = controller.optimize(postBias,velBias,jacPL,jacPR,Wimp,Wpos,Qrange,Qdotrange,tauRange)
        lossT =  controller.optimize(postBias,velBias,jacPL,jacPR,Wimp,Wpos,Qrange,Qdotrange,tauRange)
        object_pose = [list(data.body("collision_object").xpos.copy()),list(data.body("collision_object").xquat.copy())]
        dist_obj = np.linalg.norm(np.array(object_pose[0]) - np.array(final_object_pose[0]))
        ope.append(dist_obj)
        loss.append(lossT)
        KE_left = 0.5 * controller.qLdot.T @ controller.ML @ controller.qLdot 
        KE_right = 0.5 * controller.qRdot.T @ controller.MR @ controller.qRdot
        KE = (KE_left + KE_right) / 2
        # ic(KE)
        # for i, joint_name in enumerate(model.joint_names):
        #     torque = data.qfrc_actuator[i]
        #     print(f"Joint: {joint_name}, Torque: {torque}")
        
        # tausL.append(tauL.tolist())
        # tausR.append(tauR.tolist())
        # FDotL.append(slope_L)
        # FDotR.append(slope_R)
        FL.append(force_eef_L)
        FR.append(force_eef_R)
        
        # tausL.append(copy.deepcopy(data.sensor('frceef0').data))
        # tausR.append(copy.deepcopy(data.sensor('frceef1').data))
        object_pos_err.append(dist_obj)
        
        
        
    
        if(stage == 1):
            controller.gripperCtrl("open", "both")
        elif(stage == 2):
            controller.gripperCtrl("open", "both")
        elif(stage == 3 or stage == 4):
            controller.gripperCtrl("close", "both", gripper_close_amt)
            gripper_close_amt = max(gripper_close_amt - 0.001, 0)

        mujoco.mj_step(model,data)

        jacPL = controller.JL
        jacPR = controller.JR
        viewer.sync()
    
    force_name = "forces_50per"
    os.makedirs(f'./results/{force_name}', exist_ok=True)
    # loss = np.array(loss)
    # np.save(f"./results/{force_name}/loss.npy",loss)
    
    tausL = np.array(tausL)
    tausR = np.array(tausR)
    # eucl_dist = np.array(eucl_dist)
    # object_pos_err = np.array(object_pos_err)
    # FDotR = np.array(FDotR)
    # FDotL = np.array(FDotL)
    # ForceR = np.array(FR)
    # ForceL = np.array(FL)
    
    # np.save(f"./results/{force_name}/cvxpyLoss.npy", loss)
    # np.save(f"./results/{force_name}/OPE.npy", object_pos_err)
    # np.save(f"./results/{force_name}/EuclDist.npy", eucl_dist)
    # # np.save("OQE.npy", self.object_quat_err)
    # np.save("eefL_force_k20.npy", force_eef_L)
    # np.save("eefR_force_k20.npy", force_eef_R)
    np.save(f"./results/{force_name}/TsL.npy", tausL)
    np.save(f"./results/{force_name}/TsR.npy", tausR)
    # np.save(f"./results/{force_name}/FdotL.npy", FDotL)
    # np.save(f"./results/{force_name}/FdotR.npy", FDotR)
    # np.save(f"./results/{force_name}/FL_k600.npy", ForceL)
    # np.save(f"./results/{force_name}/FR_k600.npy", ForceR)

    # print("saved")

    # controller.makeplots()

if __name__ == "__main__":
    main()
