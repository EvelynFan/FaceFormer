import numpy as np
import scipy.io.wavfile as wav
import librosa
import os, sys, shutil, argparse, copy, pickle
import math, scipy
from faceformer import Faceformer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import cv2
import tempfile
from subprocess import call
from smpl_webuser.serialization import load_model

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # egl
import pyrender
from psbody.mesh import Mesh
import trimesh


@torch.no_grad()
def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # build model
    model = Faceformer(args)
    model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name))))
    model = model.to(torch.device(args.device))
    model.eval()

    template_file = os.path.join(args.dataset, args.template_path)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot, (-1, one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    temp = templates[args.subject]

    template = temp.reshape((-1))
    template = np.reshape(template, (-1, template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    prediction = model.predict(audio_feature, template, one_hot)
    prediction = prediction.squeeze()  # (seq_len, V*3)
    predicted_vertices = prediction.detach().cpu().numpy()
    flame_model_fname = "./flame/generic_model.pkl"
    predicted_vertices_frames = predicted_vertices.shape[0]
    predicted_vertices = add_eye_blink_sequences(predicted_vertices, flame_model_fname)
    print(predicted_vertices.shape)
    predicted_vertices = np.reshape(predicted_vertices, (predicted_vertices_frames, 15069))
    predicted_vertices = add_head_pose_sequences(predicted_vertices, flame_model_fname)
    np.save(os.path.join(args.result_path, test_name), predicted_vertices)


def add_eye_blink_sequences(predicted_vertices_old, flame_model_fname, num_blinks=2, blink_duration=15, ):
    # Load sequence files
    num_frames = predicted_vertices_old.shape[0]
    if num_frames == 0:
        print('No sequence meshes found')
        return

    # Load FLAME head model
    model = load_model(flame_model_fname)

    blink_exp_betas = np.array(
        [0.04676158497927314, 0.03758675711005459, -0.8504121184951298, 0.10082324210507627, -0.574142329926028,
         0.6440016589938355, 0.36403779939335984, 0.21642312586261656, 0.6754551784690193, 1.80958618462892,
         0.7790133813372259, -0.24181691256476057, 0.826280685961679, -0.013525679499256753, 1.849393698014113,
         -0.263035686247264, 0.42284248271332153, 0.10550891351425384, 0.6720993875023772, 0.41703592560736436,
         3.308019065485072, 1.3358509602858895, 1.2997143108969278, -1.2463587328652894, -1.4818961382824924,
         -0.6233880069345369, 0.26812528424728455, 0.5154889093160832, 0.6116267181402183, 0.9068826814583771,
         -0.38869613253448576, 1.3311776710005476, -0.5802565274559162, -0.7920775624092143, -1.3278601781150017,
         -1.2066425872386706, 0.34250140710360893, -0.7230686724732668, -0.6859285483325263, -1.524877347586566,
         -1.2639479212965923, -0.019294228307535275, 0.2906175769381998, -1.4082782880837976, 0.9095436721066045,
         1.6007365724960054, 2.0302381182163574, 0.5367600947801505, -0.12233184771794232, -0.506024823810769,
         2.4312326730634783, 0.5622323258974669, 0.19022395712837198, -0.7729758559103581, -1.5624233513002923,
         0.8275863297957926, 1.1661887586553132, 1.2299311381779416, -1.4146929897142397, -0.42980549225554004,
         -1.4282801579740614, 0.26172301287347266, -0.5109318114918897, -0.6399495909195524, -0.733476856285442,
         1.219652074726591, 0.08194907995352405, 0.4420398361785991, -1.184769973221183, 1.5126082924326332,
         0.4442281271081217, -0.005079477284341147, 1.764084274265486, 0.2815940264026848, 0.2898827213634057,
         -0.3686662696397026, 1.9125365942683656, 2.1801452989500274, -2.3915065327980467, 0.5794919897154226,
         -1.777680085517591, 2.9015718628823604, -2.0516886588315777, 0.4146899057365943, -0.29917763685660903,
         -0.5839240983516372, 2.1592457102697007, -0.8747902386178202, -0.5152943072876817, 0.12620001057735733,
         1.3144109838803493, -0.5027032013330108, 1.2160353388774487, 0.7543834001473375, -3.512095548974531,
         -0.9304382646186183, -0.30102930208709433, 0.9332135959962723, -0.52926196689098, 0.23509772959302958])

    step = blink_duration // 3
    blink_weights = np.hstack(
        (np.interp(np.arange(step), [0, step], [0, 1]), np.ones(step), np.interp(np.arange(step), [0, step], [1, 0])))

    frequency = num_frames // (num_blinks + 1)
    weights = np.zeros(num_frames)
    for i in range(num_blinks):
        x1 = (i + 1) * frequency - blink_duration // 2
        x2 = x1 + 3 * step
        if x1 >= 0 and x2 < weights.shape[0]:
            weights[x1:x2] = blink_weights

    predicted_vertices = np.zeros((num_frames, model.v_template.shape[0], model.v_template.shape[1]))

    for frame_idx in range(num_frames):
        model.v_template[:] = Mesh(np.reshape(predicted_vertices_old[frame_idx], (5023, 3))).v
        model.betas[300:] = weights[frame_idx] * blink_exp_betas
        predicted_vertices[frame_idx] = model.r
    return predicted_vertices


def add_head_pose_sequences(predicted_vertices_old, flame_model_fname, pose_frame=70, pose_idx=2,
                            rot_angle=np.pi / 60):
    # Load sequence files
    num_frames = predicted_vertices_old.shape[0]
    if num_frames == 0:
        print('No sequence meshes found')
        return
    # Load FLAME head model
    pose_frame_tuple = []
    model = load_model(flame_model_fname)
    model_parms = np.zeros((num_frames, model.pose.shape[0]))
    # Generate interpolated pose parameters for each frame
    pose_frame_tuple.extend((0,) * (num_frames % pose_frame))
    pose_frames = int(num_frames / pose_frame)
    for i in range(pose_frames):
        angle_range = random.randint(0, 10)
        if angle_range <= 3:
            left_rot_angle = np.pi / 180
            right_rot_angle = np.pi / 180
        else:
            left_rot_angle = np.pi / (random.randint(160, 180))
            right_rot_angle = np.pi / (random.randint(160, 180))
        x1, y1 = [0, pose_frame // 4], [0, left_rot_angle]
        x2, y2 = [pose_frame // 4, pose_frame // 2], [left_rot_angle, 0]
        x3, y3 = [pose_frame // 2, 3 * pose_frame // 4], [0, -right_rot_angle]
        x4, y4 = [3 * pose_frame // 4, pose_frame], [-right_rot_angle, 0]
        xsteps1 = np.arange(0, pose_frame // 4)
        xsteps2 = np.arange(pose_frame // 4, pose_frame // 2)
        xsteps3 = np.arange(pose_frame // 2, 3 * pose_frame // 4)
        xsteps4 = np.arange(3 * pose_frame // 4, pose_frame)
        pose_frame_tuple.extend((np.interp(xsteps1, x1, y1),
                                 np.interp(xsteps2, x2, y2),
                                 np.interp(xsteps3, x3, y3),
                                 np.interp(xsteps4, x4, y4)))
    model_parms[:, pose_idx] = np.hstack(pose_frame_tuple)

    predicted_vertices = np.zeros((num_frames, model.v_template.shape[0], model.v_template.shape[1]))
    for frame_idx in range(num_frames):
        model.v_template[:] = Mesh(np.reshape(predicted_vertices_old[frame_idx], (5023, 3))).v
        model.pose[:] = model_parms[frame_idx]
        predicted_vertices[frame_idx] = model.r
    return predicted_vertices


# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(args, mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0):
    if args.dataset == "BIWI":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
    elif args.dataset == "vocaset":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v - t_center).T).T + t_center
    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8
    )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material, smooth=True)

    if args.background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                       fy=camera_params['f'][1],
                                       cx=camera_params['c'][0],
                                       cy=camera_params['c'][1],
                                       znear=frustum['near'],
                                       zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3, 3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3, 3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]


def render_sequence(args):
    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    predicted_vertices_path = os.path.join(args.result_path, test_name + ".npy")
    if args.dataset == "BIWI":
        template_file = os.path.join(args.dataset, args.render_template_path, "BIWI.ply")
    elif args.dataset == "vocaset":
        template_file = os.path.join(args.dataset, args.render_template_path, "FLAME_sample.ply")

    print("rendering: ", test_name)

    template = Mesh(filename=template_file)
    predicted_vertices = np.load(predicted_vertices_path)
    predicted_vertices = np.reshape(predicted_vertices, (-1, args.vertice_dim // 3, 3))

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_frames = predicted_vertices.shape[0]
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)

    writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (800, 800), True)
    center = np.mean(predicted_vertices[0], axis=0)

    for i_frame in range(num_frames):
        render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        pred_img = render_mesh_helper(args, render_mesh, center)
        pred_img = pred_img.astype(np.uint8)
        writer.write(pred_img)

    writer.release()
    file_name = test_name + "_" + args.subject + "_condition_" + args.condition

    video_fname = os.path.join(output_path, file_name + '.mp4')
    cmd = ('ffmpeg' + ' -i {0} -i {1} -pix_fmt yuv420p -qscale 0 -vcodec copy -acodec aac {2} '.format(
        tmp_video_file.name, wav_path, video_fname)).split()
    call(cmd)


def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--model_name", type=str, default="biwi")
    parser.add_argument("--dataset", type=str, default="BIWI", help='vocaset or BIWI')
    parser.add_argument("--fps", type=float, default=25, help='frame rate - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=128, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--vertice_dim", type=int, default=23370 * 3,
                        help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")
    parser.add_argument("--output_path", type=str, default="demo/output", help='path of the rendered video sequence')
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions')
    parser.add_argument("--condition", type=str, default="M3", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="M1",
                        help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--render_template_path", type=str, default="templates",
                        help='path of the mesh in BIWI/FLAME topology')
    args = parser.parse_args()

    test_model(args)
    render_sequence(args)


if __name__ == "__main__":
    main()
