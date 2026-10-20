"""Parametric Blender 4.5 build of the automated RGB+NIR mango capture rig.

Single source of truth for the geometry, the design checks, the printed-part STLs,
the MDF cut list, the renders and the capture-cycle animation.

Usage (everything after ``--`` is parsed by this script)::

    blender -b --factory-startup --python hardware/capture_rig/build_capture_rig.py -- --check-only
    blender -b --factory-startup --python hardware/capture_rig/build_capture_rig.py -- --export-dir hardware/capture_rig
    blender -b --factory-startup --python hardware/capture_rig/build_capture_rig.py -- --export-dir DIR --no-render

Without arguments (for example when exec'd inside the Blender GUI) only the scene is built.
Other options: ``--only blend,stl,csv,stills,anim`` restricts the export steps and ``--skip-build``
runs the checks on an empty scene (they must all fail; the pytest wrapper relies on that).

Conventions
-----------
* Every dimension below is in millimetres. One Blender unit is one metre
  (unit display is millimetres) and STLs are exported with ``global_scale=1000``.
* Origin: centre of the INTERNAL floor. X runs along the spindle axis (left to
  right seen from the front), Y front (-) to back (+), Z up.
* Printed parts carry the custom property ``printed=True`` and also live in the
  PRINTED collection. Other tags used by the checks: ``rotating``, ``los_ignore``,
  ``outside_frame``, ``envelope_exempt``, ``variant`` (``jig`` objects are hidden
  by default) and ``scene_aux`` (render helpers that are not part of the rig).
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import bpy
    import bmesh
    from mathutils import Euler, Matrix, Vector
    from mathutils.bvhtree import BVHTree
    import numpy as np
except ImportError:  # pragma: no cover - the script only runs inside Blender
    sys.stderr.write("This script must run inside Blender (blender -b --python ...)\n")
    raise

# =============================================================================
# CONSTANTS (millimetres unless stated)
# =============================================================================
MM = 0.001  # millimetres -> Blender units (metres)

# ---- Enclosure ---------------------------------------------------------------
MDF_T = 9.0
INT_X, INT_Y, INT_Z = 420.0, 340.0, 380.0
EXT_X = INT_X + 2 * MDF_T            # 438 (floor and top panel)
EXT_Y = INT_Y + 2 * MDF_T            # 358
ENV_HALF_X, ENV_HALF_Y = EXT_X / 2, EXT_Y / 2
ENV_Z0, ENV_Z1 = -MDF_T, INT_Z + MDF_T
TOP_OPEN_X, TOP_OPEN_Y = 90.0, 60.0
TOP_OPEN_CX, TOP_OPEN_CY = -30.0, 0.0
CABLE_HOLE_D = 12.0
CABLE_HOLE_XS = (-150.0, 150.0)
CABLE_HOLE_Z = 60.0
LINER_T = 3.0                        # white PVC foam board
BLACK_BAND_H = 130.0                 # matte black below this height
DOOR_OPEN_DEG = 105.0
DOOR_CLEARANCE_NOTE = 1.0            # cut the real door this much smaller on each side

# ---- Spindle: common ---------------------------------------------------------
AXIS_Y, AXIS_Z = 0.0, 110.0
HEAD_MOUTH_X = -95.0                 # head cup mouth plane
TAIL_MOUTH_NOMINAL_X = 29.0          # tail cup mouth plane with the nominal fruit
TAIL_MOUTH_MIN_X, TAIL_MOUTH_MAX_X = -15.0, 80.0
FRUIT_LEN_MIN, FRUIT_LEN_MAX = 80.0, 175.0
FRUIT_CLEARANCE_R = 55.0             # largest fruit radius about the axis

# ---- Headstock ---------------------------------------------------------------
MOTOR_BODY, MOTOR_LEN = 42.3, 40.0
MOTOR_FACE_X = -150.0
MOTOR_BOSS_D, MOTOR_BOSS_L = 22.0, 2.0
MOTOR_SHAFT_D, MOTOR_SHAFT_L = 5.0, 24.0
MOTOR_HOLE_PITCH = 31.0
BRACKET_BOSS_HOLE_D = 22.5
BRACKET_M3_D = 3.2
BRACKET_PLATE_T = 6.0
BRACKET_W = 56.0
BRACKET_TOP_Z = 138.0
BRACKET_FOOT_X0, BRACKET_FOOT_X1 = -190.0, -130.0
BRACKET_FOOT_W, BRACKET_FOOT_T = 70.0, 5.0
FLOOR_SCREW_D = 4.5
POINTER_X1 = -97.0                   # pointer tip stays outside the fruit span
POINTER_W, POINTER_T = 3.0, 2.0
HUB_X0, HUB_X1, HUB_D = -143.0, -121.0, 20.0
HUB_BORE_D, HUB_BORE_DEPTH, GRUB_D = 5.0, 20.0, 3.0
SPIGOT_D, SPIGOT_L = 16.0, 8.0
CUP_MOUTH_D, CUP_RIM_OD, CUP_LEN = 36.0, 40.0, 26.0
CUP_STEM_OD = 20.0
CUP_CAVITY_FLOOR_D, CUP_CAVITY_DEPTH = 12.0, 15.0
CUP_SOCKET_D, CUP_SOCKET_DEPTH = 16.0, 8.0
CUP_NOTCH_W, CUP_NOTCH_DEPTH = 3.0, 4.0
CUP_LABEL_H = 4.0

# ---- Tailstock ---------------------------------------------------------------
ROD_D, ROD_Y, ROD_Z = 8.0, 25.0, 18.0
ROD_X0, ROD_X1 = -40.0, 200.0
HOLDER_L_X = (-40.0, -28.0)
HOLDER_R_X = (188.0, 200.0)
HOLDER_Y, HOLDER_Z = 36.0, 36.0
HOLDER_FOOT_T = 5.0
HOLDER_FOOT_L_X1 = -10.0
HOLDER_FOOT_R_X0 = 170.0
ROD_HOLE_D = 8.2
LM8UU_OD, LM8UU_L = 15.0, 24.0
BUSHING_POCKET_D = 15.2
BEARING_OD, BEARING_ID, BEARING_W = 22.0, 8.0, 7.0   # 608
HOUSING_X = (59.0, 70.0)
HOUSING_D = 34.0
HOUSING_WALL = 4.0
BEARING_RECESS_D = 22.2
HOUSING_FRONT_HOLE_D = 14.0
CARRIAGE_BX = 64.5                   # housing centre X at the nominal fruit
CARRIAGE_BLOCK_HALF_X = 14.0
CARRIAGE_BLOCK_Y = 36.0
CARRIAGE_BLOCK_Z = (7.0, 29.0)
CARRIAGE_RIB_HALF_Y = 14.0
CARRIAGE_GUSSET_LEN = 35.0
TAIL_HUB_FLANGE = (55.0, 58.0)
TAIL_HUB_FLANGE_D = 26.0
TAIL_HUB_SPACER_D = 12.0
TAIL_HUB_SPACER_X1 = 63.0
BOLT_D = 8.0
BOLT_HEAD = (47.7, 53.0)
BOLT_HEAD_AF = 13.0
BOLT_END_X = 78.0
NUT = (70.0, 76.5)
SPRING_WIRE_D, SPRING_OD, SPRING_TURNS = 1.2, 11.0, 15
KNOB_POST_Y = -31.0
KNOB_POST_TOP_Z = 70.0
KNOB_D, KNOB_H = 28.0, 14.0

# ---- Mango (nominal) ---------------------------------------------------------
MANGO_SEMI = (62.0, 45.0, 42.0)
MANGO_CX = (HEAD_MOUTH_X + TAIL_MOUTH_NOMINAL_X) / 2   # -33
MANGO_NUB = 4.0                      # stem scar / apex beak seated inside each cup

# ---- Cameras -----------------------------------------------------------------
CAM_LENS_Z = 345.0
CAM_NIR_X, CAM_RGB_X, CAM_Y = -46.5, -13.5, 0.0
SENSOR_W_MM, FOCAL_MM = 3.68, 3.04
RES_X, RES_Y = 1280, 720
CAL_PLANE_Z = 150.0
FOOTPRINT_X, FOOTPRINT_Y = 175.0, 110.0
FOOTPRINT_CX = -30.0
TOWER_CHEEK_Y = (25.0, 29.0)
TOWER_CHEEK_X = (-72.0, 12.0)
TOWER_BOTTOM_Z = 282.0          # window 272..292: lower enters the camera frame, higher lets LED light reach the lenses
TOWER_FLANGE_X = (-85.0, 25.0)
TOWER_FLANGE_Y = 40.0
TOWER_FLANGE_T = 4.0
TOWER_OPEN_X, TOWER_OPEN_Y = (-68.0, 8.0), 22.0
TOWER_SLOT_XS = (-55.0, -5.0)
TOWER_SLOT_ZC, TOWER_SLOT_TRAVEL, TOWER_SLOT_W = 341.0, 20.0, 4.5   # +-10 mm: lens Z 335..355 (WD 185..205)
# Wider travel fails check (k): lower lets LED light reach the lenses (and the 110 mm fruit leaves
# the frame at WD 165), higher puts the tower cheeks inside the camera frame.
CARRIER_PLATE_Z = (352.0, 356.0)
CARRIER_PLATE_X = (-68.0, 8.0)
CARRIER_PLATE_Y = (-24.0, 25.0)
CARRIER_TAB_Y = (21.0, 25.0)
CARRIER_TAB_X = (-62.0, 2.0)
CARRIER_TAB_Z0 = 330.0
NOIR_BOARD = (25.0, 24.0, 1.0)       # X, Y, thickness
NOIR_HOLES = (21.0, 12.5)
NOIR_BARREL_D = 8.5
ARDUCAM_BOARD = (38.0, 38.0, 1.6)
ARDUCAM_BARREL_D = 14.0
ARDUCAM_BOARD_Z0 = 359.0
NOIR_BORE_D, ARDUCAM_BORE_D = 14.0, 22.0

# ---- Lighting ----------------------------------------------------------------
BAR_X0, BAR_X1 = -130.0, 70.0        # 200 mm printed stand
BAR_CX = -30.0
BAR_FACE_CY, BAR_FACE_CZ = 135.0, 75.0
BAR_TILT_DEG = 35.0                  # LED normal tilt from vertical, towards the wall
BAR_FACE_W, BAR_PLATE_T = 76.0, 4.0
BAR_PLATE_X = (-120.0, 60.0)
BAR_FIN_D0, BAR_FIN_T = 101.2, 3.0   # fin inner face distance from the axis plane, thickness
BAR_FIN_TOP_Z = 118.0
BAR_FOOT_T = 3.0
BAR_FOOT_D = (122.0, 168.0)      # foot starts outside the frame footprint on the floor (|Y| > 117.5)
BAR_FIN_Z0 = 62.0                # fin bottom: lower, it would enter the camera frame (limit ~48)
BAR_RIB_FLOOR_D = 121.0
BAR_FIN_X = (-129.0, 69.0)
BAR_RIB_T = 4.0
BAR_RIB_XS = (-118.0, -74.0, -30.0, 14.0, 58.0)
BAR_SCREW_XS = (-125.0, 65.0)
RING_D, RING_T = 70.0, 3.0
RING_LED_RINGS = ((24, 31.0), (24, 23.5))   # (count, radius) -> 48 LEDs
PANEL_X_OFFSETS = (-60.0, 60.0)
PANEL_ALONG_X, PANEL_ACROSS, PANEL_T = 40.0, 52.0, 2.0
PANEL_GRID = (4, 9)                  # rows along X, columns across -> 36 LEDs
LED_DOME_D, LED_DOME_H = 4.5, 1.5

# ---- Electronics -------------------------------------------------------------
DECK_X = (-125.0, 65.0)
DECK_Y = (48.0, 178.0)
DECK_H, DECK_WALL, DECK_FLOOR_T = 35.0, 3.0, 3.0
DECK_POST_H, DECK_POST_D = 5.0, 6.0
PI_SIZE, PI_HOLES = (65.0, 30.0), (58.0, 23.0)
RELAY_SIZE = (50.0, 40.0)
DRIVER_SIZE = (50.0, 40.0)
BUCK_SIZE = (43.0, 21.0)
TERMINAL_SIZE = (30.0, 12.0)
BUTTON_D, LED_STATUS_D = 16.0, 5.0
BUTTON_PANEL_X = (100.0, 146.0)
BUTTON_PANEL_Y = (-178.0, -156.0)
BUTTON_PANEL_H = 18.0
JACK_X, JACK_Z = 150.0, 120.0
JACK_HOUSING = (30.0, 24.0, 22.0)    # X, Y (outward), Z
SWITCH_POS = (200.0, -158.0, 360.0)

# ---- Calibration jig ---------------------------------------------------------
JIG_PLATE = (140.0, 90.0, 3.0)
JIG_PLATE_TOP_Z = CAL_PLANE_Z
JIG_PLUG_LEN = 8.0
JIG_BAR_Y, JIG_BAR_Z = 32.0, 12.0
JIG_POST_XS = (-73.5, -19.5)       # centred on the white gaps between ArUco marker columns
JIG_POST_YS = (-13.5, 13.5)        # ... and between marker rows, so the M3 holes never cut a marker

# ---- Print / export ----------------------------------------------------------
PRINT_VOLUME = 200.0
BBOX_TOL = 0.01
ARUCO_DICT = "DICT_4X4_50"
ARUCO_FILE = "calibration/aruco_board_140x90mm.jpg"
REPO_VENV_PY = "/home/rsphobos/PROYECTOS_GABO/MCP_LangGraph_Project/.venv/bin/python"

COLLECTION_NAMES = ("ENCLOSURE", "LINERS", "SPINDLE", "MANGO", "CAMERAS", "LIGHTING",
                    "ELECTRONICS", "CALIBRATION_JIG", "PRINTED")

# ---- Capture-cycle animation (frames @ 24 fps) -------------------------------
FPS = 24
FRAME_END = 240
DOOR_CLOSE = (1, 24)                 # door swings shut
FACE_START = (25, 73, 121, 169)      # first frame of the window of faces 1..4
FACE_LEN = 48
DOOR_OPEN_AT = (217, 240)
W_ON, W_OFF = 2, 14                  # white LEDs on/off, offset inside the face window
BLINK_RGB = (8, 11)                  # status LED blink during the RGB shot
N_ON, N_OFF = 18, 30                 # 850 nm LEDs
BLINK_NIR = (24, 27)                 # status LED blink during the NIR shot
ROT_SPAN = (33, 45)                  # spindle turns 90 deg
HOLD_RGB, HOLD_NIR = 9, 25           # window offsets used for the still renders
STEPS_PER_FACE = 800                 # 3200 steps/rev (1/16 microstep) / 4
WHITE_PEAK_W, NIR_PEAK_W = 1.2, 1.0       # Blender area-light watts (preview units, tuned for AgX)
EXPOSURE_RGB, EXPOSURE_NIR = -1.0, -1.6   # stops, camera stills only
BOUNCE_PEAK_W = 10.0                 # animation-only fill that stands in for the ceiling bounce (EEVEE has no GI)
ANIM_AMBIENT = 0.25                  # world strength during the animation so the idle rig stays visible
EMISSION_PEAK = {"LED_WHITE": 40.0, "LED_NIR": 25.0, "LED_STATUS": 30.0}

# <<CONSTANTS_END>>

# =============================================================================
# Runtime registry (filled by the builders, read by the checks)
# =============================================================================
class Rig:
    led_emitters = []        # dicts: kind, bar, pos (Vector, metres)
    deck_rects = []          # (name, x0, y0, x1, y1) in deck-local mm
    print_orient = {}        # part name -> Euler (radians) used when exporting the STL
    materials = {}
    root = Path(globals().get("__file__", "hardware/capture_rig/build_capture_rig.py")).resolve().parent


# =============================================================================
# Scene helpers
# =============================================================================
def clear_scene():
    """Remove every object and datablock so the build starts from nothing."""
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for attr in ("meshes", "curves", "lights", "cameras", "materials", "images", "collections",
                 "actions", "node_groups", "worlds"):
        coll = getattr(bpy.data, attr)
        for block in list(coll):
            try:
                coll.remove(block)
            except RuntimeError:
                pass
    Rig.led_emitters = []
    Rig.deck_rects = []
    Rig.print_orient = {}
    Rig.materials = {}


# =============================================================================
# Geometry helpers (inputs in mm, meshes are created in metres)
# =============================================================================
def V(x, y, z):
    return Vector((x * MM, y * MM, z * MM))


def coll(name):
    return bpy.data.collections[name]


def make_collections():
    root = bpy.context.scene.collection
    for n in COLLECTION_NAMES:
        if n not in bpy.data.collections:
            root.children.link(bpy.data.collections.new(n))


def bm_xform(bm, matrix):
    bmesh.ops.transform(bm, matrix=matrix, verts=bm.verts)
    return bm


def bm_box(x0, x1, y0, y1, z0, z1):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x = (x0 if v.co.x < 0 else x1) * MM
        v.co.y = (y0 if v.co.y < 0 else y1) * MM
        v.co.z = (z0 if v.co.z < 0 else z1) * MM
    return bm


def _axis_matrix(axis, c):
    t = Matrix.Translation(V(*c))
    if axis == "x":
        return t @ Matrix.Rotation(math.pi / 2, 4, "Y")
    if axis == "y":
        return t @ Matrix.Rotation(-math.pi / 2, 4, "X")
    return t


def bm_cyl(axis, c, r, length, seg=32, r2=None):
    """Cylinder/frustum along ``axis`` centred on ``c``; ``r`` is the low-coordinate end, ``r2`` the high end."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r * MM,
                          radius2=(r if r2 is None else r2) * MM, depth=length * MM, matrix=_axis_matrix(axis, c))
    return bm


def bm_cyl_x(x0, x1, r0, r1=None, y=AXIS_Y, z=AXIS_Z, seg=32):
    return bm_cyl("x", ((x0 + x1) / 2, y, z), r0, x1 - x0, seg, r1)


def bm_cyl_z(x, y, z0, z1, r, seg=32):
    return bm_cyl("z", (x, y, (z0 + z1) / 2), r, z1 - z0, seg)


def bm_cyl_y(x, y0, y1, z, r, seg=32):
    return bm_cyl("y", (x, (y0 + y1) / 2, z), r, y1 - y0, seg)


def bm_prism(plane, pts, a0, a1):
    """Polygon ``pts`` in ``plane`` ('yz' | 'xz' | 'xy') extruded between a0 and a1 on the remaining axis."""
    def p3(p, a):
        if plane == "yz":
            return Vector((a, p[0], p[1])) * MM
        if plane == "xz":
            return Vector((p[0], a, p[1])) * MM
        return Vector((p[0], p[1], a)) * MM
    bm = bmesh.new()
    lo = [bm.verts.new(p3(p, a0)) for p in pts]
    hi = [bm.verts.new(p3(p, a1)) for p in pts]
    caps = [bm.faces.new(lo), bm.faces.new(hi)]
    n = len(pts)
    for i in range(n):
        bm.faces.new((lo[i], lo[(i + 1) % n], hi[(i + 1) % n], hi[i]))
    bmesh.ops.triangulate(bm, faces=caps, ngon_method="EAR_CLIP")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def bm_tube(axis, c, ro, ri, length, seg=32):
    """Hollow tube (manifold) along ``axis``."""
    bm = bmesh.new()
    rings = []
    for r, z in ((ro, -0.5), (ro, 0.5), (ri, 0.5), (ri, -0.5)):
        rings.append([bm.verts.new(Vector((r * math.cos(2 * math.pi * i / seg) * MM,
                                           r * math.sin(2 * math.pi * i / seg) * MM, z * length * MM)))
                      for i in range(seg)])
    for k in range(4):
        a, b = rings[k], rings[(k + 1) % 4]
        for i in range(seg):
            bm.faces.new((a[i], a[(i + 1) % seg], b[(i + 1) % seg], b[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm_xform(bm, _axis_matrix(axis, c))


def bm_hex_x(x0, x1, across_flats, y=AXIS_Y, z=AXIS_Z):
    return bm_cyl_x(x0, x1, across_flats / math.sqrt(3), None, y, z, seg=6)


def bm_sphere(c, radius_xyz, u=48, v=24):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=1.0)
    for vt in bm.verts:
        vt.co = Vector((vt.co.x * radius_xyz[0], vt.co.y * radius_xyz[1], vt.co.z * radius_xyz[2])) * MM + V(*c)
    return bm


def bm_helix_tube(axis, x0, length, r_mean, wire_d, turns, c, seg_per_turn=14, wire_seg=6):
    """Coil spring surrounding a line parallel to ``axis`` at ``c`` (centre of the first turn)."""
    bm = bmesh.new()
    n = int(turns * seg_per_turn)
    rings = []
    for i in range(n + 1):
        t = i / n
        ang = 2 * math.pi * turns * t
        centre = Vector((x0 + length * t, r_mean * math.cos(ang), r_mean * math.sin(ang)))
        radial = Vector((0, math.cos(ang), math.sin(ang)))
        tang = Vector((length / (2 * math.pi * turns), -r_mean * math.sin(ang), r_mean * math.cos(ang))).normalized()
        side = tang.cross(radial).normalized()
        ring = []
        for k in range(wire_seg):
            a = 2 * math.pi * k / wire_seg
            p = centre + (radial * math.cos(a) + side * math.sin(a)) * (wire_d / 2)
            ring.append(bm.verts.new(p * MM))
        rings.append(ring)
    for i in range(n):
        for k in range(wire_seg):
            bm.faces.new((rings[i][k], rings[i][(k + 1) % wire_seg], rings[i + 1][(k + 1) % wire_seg], rings[i + 1][k]))
    bm.faces.new(rings[0][::-1])
    bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm_xform(bm, Matrix.Translation(Vector((0, c[1], c[2])) * MM))


# ---- objects -----------------------------------------------------------------
def new_obj(name, bm, colls, material=None, **props):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.update()
    ob = bpy.data.objects.new(name, me)
    for c in ([colls] if isinstance(colls, str) else colls):
        coll(c).objects.link(ob)
    mats = material if isinstance(material, (list, tuple)) else ([material] if material else [])
    for m in mats:
        ob.data.materials.append(m)
    for k, v in props.items():
        ob[k] = v
    for poly in me.polygons:
        poly.use_smooth = False
    return ob


def smooth(ob, angle_deg=40.0):
    for p in ob.data.polygons:
        p.use_smooth = True
    ob.data.update()
    return ob


def parent_keep(child, parent):
    """Parent without moving: the child's mesh coordinates are already world coordinates."""
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()


def _apply_modifiers(ob):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    bpy.data.meshes.remove(old)


def _boolean(base, operands, operation):
    if not operands:
        return
    tmp = bpy.data.collections.new("TMP_CSG")
    bpy.context.scene.collection.children.link(tmp)
    objs = []
    for i, bm in enumerate(operands):
        me = bpy.data.meshes.new(f"tmp_{i}")
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(f"tmp_{i}", me)
        tmp.objects.link(ob)
        objs.append(ob)
    mod = base.modifiers.new("csg", "BOOLEAN")
    mod.operation = operation
    mod.operand_type = "COLLECTION"
    mod.collection = tmp
    mod.solver = "EXACT"
    mod.use_self = True
    _apply_modifiers(base)
    for ob in objs:
        me = ob.data
        bpy.data.objects.remove(ob)
        bpy.data.meshes.remove(me)
    bpy.context.scene.collection.children.unlink(tmp)
    bpy.data.collections.remove(tmp)


def cleanup_mesh(ob, merge_dist=1e-7):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-9, edges=bm.edges)
    loose = [v for v in bm.verts if not v.link_edges]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def csg_solid(name, adds, cuts, colls, material=None, **props):
    """Union of ``adds`` minus ``cuts`` (lists of bmeshes in metres) as one mesh object."""
    ob = new_obj(name, adds[0], colls, material, **props)
    _boolean(ob, adds[1:], "UNION")
    _boolean(ob, cuts, "DIFFERENCE")
    cleanup_mesh(ob)
    return ob


def printed(ob, orient_deg=(0, 0, 0), **props):
    """Tag ``ob`` as a printed part and register its print orientation (degrees XYZ)."""
    ob["printed"] = True
    for k, v in props.items():
        ob[k] = v
    if ob.name not in coll("PRINTED").objects:
        coll("PRINTED").objects.link(ob)
    Rig.print_orient[ob.name] = Euler([math.radians(a) for a in orient_deg], "XYZ")
    return ob


def paint_inner_face(ob, inward):
    """Assign material slot 1 (black paint) to faces whose normal points along ``inward``."""
    inward = Vector(inward).normalized()
    for p in ob.data.polygons:
        p.material_index = 1 if p.normal.dot(inward) > 0.9 else 0


# ---- materials ---------------------------------------------------------------
def make_material(name, color, rough=0.6, metallic=0.0, emission=None, spec=0.4):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Specular IOR Level"].default_value = spec
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission[0], 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission[1]
    m.diffuse_color = (*color, 1.0)
    Rig.materials[name] = m
    return m


def build_materials():
    mk = make_material
    mk("MDF_TAN", (0.50, 0.36, 0.22), 0.85)
    mk("MDF_BLACK", (0.012, 0.012, 0.012), 0.95, spec=0.1)
    mk("WHITE_PVC", (0.86, 0.86, 0.84), 0.75, spec=0.2)
    mk("PETG_BLACK", (0.022, 0.022, 0.024), 0.55, spec=0.3)
    mk("TPU_BLACK", (0.03, 0.03, 0.03), 0.85, spec=0.15)
    mk("STEEL", (0.55, 0.56, 0.58), 0.3, 1.0)
    mk("MOTOR_GREY", (0.08, 0.08, 0.09), 0.45, 0.8)
    mk("PCB_GREEN", (0.02, 0.22, 0.08), 0.5)
    mk("PCB_BLUE", (0.03, 0.10, 0.45), 0.5)
    mk("PCB_RED", (0.35, 0.03, 0.03), 0.5)
    mk("BLACK_OXIDE", (0.02, 0.02, 0.02), 0.5, 0.6)
    mk("LABEL_WHITE", (0.9, 0.9, 0.9), 0.6)
    mk("RUBBER", (0.02, 0.02, 0.02), 0.9)
    mk("BRASS", (0.75, 0.55, 0.2), 0.35, 1.0)
    mk("HOUSING_GREY", (0.12, 0.12, 0.13), 0.5, 0.2)
    for nm, col in (("LED_WHITE", (1.0, 0.96, 0.88)), ("LED_NIR", (0.9, 0.04, 0.02)), ("LED_STATUS", (0.1, 1.0, 0.2))):
        m = mk(nm, col, 0.3, emission=(col, 0.0))
        m.diffuse_color = (*col, 1.0)
    build_mango_material()
    build_aruco_material()


def build_mango_material():
    m = bpy.data.materials.new("MANGO")
    m.use_nodes = True
    nt = m.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.42
    bsdf.inputs["Specular IOR Level"].default_value = 0.45
    bsdf.inputs["Subsurface Weight"].default_value = 0.08
    bsdf.inputs["Subsurface Radius"].default_value = (0.5, 0.25, 0.1)
    coord = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coord.outputs["Object"], sep.inputs["Vector"])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 55.0
    noise.inputs["Detail"].default_value = 6.0
    links.new(coord.outputs["Object"], noise.inputs["Vector"])

    def math_node(op, a=None, b=None, clamp=False):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        if a is not None:
            if isinstance(a, (int, float)):
                n.inputs[0].default_value = a
            else:
                links.new(a, n.inputs[0])
        if b is not None:
            if isinstance(b, (int, float)):
                n.inputs[1].default_value = b
            else:
                links.new(b, n.inputs[1])
        return n.outputs[0]

    # red blush towards +Y/+Z with noisy edge
    ny = math_node("MULTIPLY", sep.outputs["Y"], 0.5 / (MANGO_SEMI[1] * MM))
    nz = math_node("MULTIPLY", sep.outputs["Z"], 0.35 / (MANGO_SEMI[2] * MM))
    nn = math_node("MULTIPLY", math_node("SUBTRACT", noise.outputs["Fac"], 0.5), 0.9)
    s = math_node("ADD", math_node("ADD", ny, nz), nn)
    blush = math_node("MULTIPLY", math_node("SUBTRACT", s, 0.05), 1.4, clamp=True)
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.86, 0.56, 0.04, 1)
    ramp.color_ramp.elements[1].color = (0.62, 0.08, 0.025, 1)
    mid = ramp.color_ramp.elements.new(0.5)
    mid.color = (0.90, 0.38, 0.035, 1)
    links.new(blush, ramp.inputs["Fac"])
    # one subtle darker bruise patch on the top face
    px, py, pz = 0.006, 0.004, MANGO_SEMI[2] * MM
    dvec = nodes.new("ShaderNodeVectorMath")
    dvec.operation = "DISTANCE"
    links.new(coord.outputs["Object"], dvec.inputs[0])
    dvec.inputs[1].default_value = (px, py, pz)
    wob = math_node("MULTIPLY", math_node("SUBTRACT", noise.outputs["Fac"], 0.5), 0.006)
    dist = math_node("ADD", dvec.outputs["Value"], wob)
    mask = nodes.new("ShaderNodeMapRange")
    mask.clamp = True
    mask.inputs["From Min"].default_value = 0.016
    mask.inputs["From Max"].default_value = 0.005
    links.new(dist, mask.inputs["Value"])
    mix = nodes.new("ShaderNodeMixRGB")
    mix.inputs["Color2"].default_value = (0.20, 0.09, 0.03, 1)
    links.new(ramp.outputs["Color"], mix.inputs["Color1"])
    amount = math_node("MULTIPLY", mask.outputs["Result"], 0.55)
    links.new(amount, mix.inputs["Fac"])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    Rig.materials["MANGO"] = m


def build_aruco_material():
    path = Rig.root / ARUCO_FILE
    m = bpy.data.materials.new("ARUCO_BOARD")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.8
    tex = nt.nodes.new("ShaderNodeTexImage")
    if path.exists():
        tex.image = bpy.data.images.load(str(path))
    else:  # fallback: procedural checker (cv2.aruco board not generated yet)
        chk = nt.nodes.new("ShaderNodeTexChecker")
        chk.inputs["Scale"].default_value = 14.0
        nt.links.new(chk.outputs["Color"], bsdf.inputs["Base Color"])
        Rig.materials["ARUCO_BOARD"] = m
        return
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / (JIG_PLATE[0] * MM), 1 / (JIG_PLATE[1] * MM), 1)
    mp.inputs["Location"].default_value = (0.5, 0.5, 0)
    nt.links.new(coord.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    Rig.materials["ARUCO_BOARD"] = m


def M(name):
    return Rig.materials[name]


# <<HELPERS>>

def build_scene():
    clear_scene()
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 1.0
    sc.unit_settings.length_unit = "MILLIMETERS"
    sc.render.resolution_x, sc.render.resolution_y = RES_X, RES_Y
    make_collections()
    build_materials()
    build_enclosure()
    build_spindle()
    build_cameras()
    build_lighting()
    build_electronics()
    build_calibration_jig(bpy.data.objects["SPINDLE_ROT"])
    build_world_and_aux()
    build_animation()
    # <<BUILD_CALLS>>


# =============================================================================
# ENCLOSURE + LINERS
# =============================================================================
def build_enclosure():
    hx, hy = ENV_HALF_X, ENV_HALF_Y
    ix, iy = INT_X / 2, INT_Y / 2
    mats = [M("MDF_TAN"), M("MDF_BLACK")]
    floor = new_obj("FLOOR", bm_box(-hx, hx, -hy, hy, -MDF_T, 0), "ENCLOSURE", mats, mdf=True)
    paint_inner_face(floor, (0, 0, 1))
    top = csg_solid("TOP_PANEL", [bm_box(-hx, hx, -hy, hy, INT_Z, INT_Z + MDF_T)],
                    [bm_box(TOP_OPEN_CX - TOP_OPEN_X / 2, TOP_OPEN_CX + TOP_OPEN_X / 2,
                            TOP_OPEN_CY - TOP_OPEN_Y / 2, TOP_OPEN_CY + TOP_OPEN_Y / 2, INT_Z - 1, INT_Z + MDF_T + 1)],
                    "ENCLOSURE", mats, mdf=True, opening=f"{TOP_OPEN_X:g}x{TOP_OPEN_Y:g} at X={TOP_OPEN_CX:g},Y={TOP_OPEN_CY:g}")
    paint_inner_face(top, (0, 0, -1))
    wl = new_obj("WALL_L", bm_box(-hx, -ix, -hy, hy, 0, INT_Z), "ENCLOSURE", mats, mdf=True)
    paint_inner_face(wl, (1, 0, 0))
    wr = new_obj("WALL_R", bm_box(ix, hx, -hy, hy, 0, INT_Z), "ENCLOSURE", mats, mdf=True)
    paint_inner_face(wr, (-1, 0, 0))
    holes = [bm_cyl_y(x, iy - 1, hy + 1, CABLE_HOLE_Z, CABLE_HOLE_D / 2, 24) for x in CABLE_HOLE_XS]
    back = csg_solid("WALL_BACK", [bm_box(-ix, ix, iy, hy, 0, INT_Z)], holes, "ENCLOSURE", mats, mdf=True,
                     holes=f"2x D{CABLE_HOLE_D:g} at X={CABLE_HOLE_XS[0]:g}/{CABLE_HOLE_XS[1]:g}, Z={CABLE_HOLE_Z:g}")
    paint_inner_face(back, (0, -1, 0))
    # door: separate object, origin on the hinge axis (left vertical edge, outer face)
    door = new_obj("DOOR", bm_box(0, INT_X, 0, MDF_T, 0, INT_Z), "ENCLOSURE", mats, mdf=True,
                   envelope_exempt=False, door_part=True)
    paint_inner_face(door, (0, 1, 0))
    door.location = V(-ix, -hy, 0)
    bpy.context.view_layer.update()

    def door_child(name, bm, material, colls, **props):
        ob = new_obj(name, bm, colls, material, door_part=True, **props)
        ob.parent = door
        return ob

    # white PVC foam liners (upper walls, ceiling) - thin separate objects
    liner = M("WHITE_PVC")
    z0, z1 = BLACK_BAND_H, INT_Z - LINER_T
    new_obj("LINER_L", bm_box(-ix, -ix + LINER_T, -iy + LINER_T, iy - LINER_T, z0, z1), "LINERS", liner, pvc=True)
    new_obj("LINER_R", bm_box(ix - LINER_T, ix, -iy + LINER_T, iy - LINER_T, z0, z1), "LINERS", liner, pvc=True)
    new_obj("LINER_BACK", bm_box(-ix + LINER_T, ix - LINER_T, iy - LINER_T, iy, z0, z1), "LINERS", liner, pvc=True)
    csg_solid("LINER_CEILING", [bm_box(-ix, ix, -iy, iy, INT_Z - LINER_T, INT_Z)],
              [bm_box(TOP_OPEN_CX - TOP_OPEN_X / 2, TOP_OPEN_CX + TOP_OPEN_X / 2, TOP_OPEN_CY - TOP_OPEN_Y / 2,
                      TOP_OPEN_CY + TOP_OPEN_Y / 2, INT_Z - LINER_T - 1, INT_Z + 1)],
              "LINERS", liner, pvc=True, opening=f"{TOP_OPEN_X:g}x{TOP_OPEN_Y:g}")
    door_child("LINER_DOOR", bm_box(LINER_T, INT_X - LINER_T, MDF_T, MDF_T + LINER_T, z0, z1), liner, "LINERS", pvc=True)

    # hinges (two barrels on the hinge axis + leaves) and the printed pull handle
    steel = M("STEEL")
    for i, (hz0, hz1) in enumerate(((40.0, 80.0), (300.0, 340.0)), start=1):
        door_child(f"HINGE_BARREL_{i}", bm_cyl_z(0, 0, hz0, hz1, 4.0, 16), steel, "ENCLOSURE", envelope_exempt=True)
        door_child(f"HINGE_LEAF_DOOR_{i}", bm_box(0, 22, -1.5, 0, hz0, hz1), steel, "ENCLOSURE", envelope_exempt=True)
        new_obj(f"HINGE_LEAF_WALL_{i}", bm_box(-hx, -ix, -hy - 1.5, -hy, hz0, hz1), "ENCLOSURE", steel, envelope_exempt=True)
    hx0 = INT_X - 45.0
    handle = csg_solid("DOOR_HANDLE",
                       [bm_box(hx0 - 7, hx0 + 7, -22, -12, 150, 250),
                        bm_box(hx0 - 5, hx0 + 5, -12.5, 0.5, 160, 170),
                        bm_box(hx0 - 5, hx0 + 5, -12.5, 0.5, 230, 240)],
                       [bm_cyl_y(hx0, -23, -17, 165, 1.5, 12), bm_cyl_y(hx0, -23, -17, 235, 1.5, 12)],
                       "ENCLOSURE", M("PETG_BLACK"), envelope_exempt=True, door_part=True)
    handle.parent = door
    printed(handle, (90, 0, 0), material="Black PETG", infill="25%")


# =============================================================================
# SPINDLE: headstock, tailstock, mango
# =============================================================================
SPIN_STEP_DEG = -90.0   # sign chosen so labels 1 -> 2 -> 3 -> 4 reach the top in order


def bm_join(*bms):
    out = bmesh.new()
    for b in bms:
        me = bpy.data.meshes.new("tmp_join")
        b.to_mesh(me)
        b.free()
        out.from_mesh(me)
        bpy.data.meshes.remove(me)
    return out


def bm_rot_axis(bm, deg):
    """Rotate ``bm`` about the spindle axis (line along X through Y=AXIS_Y, Z=AXIS_Z)."""
    pivot = Matrix.Translation(V(0, AXIS_Y, AXIS_Z))
    return bm_xform(bm, pivot @ Matrix.Rotation(math.radians(deg), 4, "X") @ pivot.inverted())


def _cup(name, mouth_x, direction):
    """TPU cup; ``direction`` = +1 when the cup body extends towards +X (tail cup), -1 for the head cup."""
    def x(d):  # distance behind the mouth plane -> X
        return mouth_x + direction * d

    def span(d0, d1, r0, r1=None, seg=48):
        a, b = x(d0), x(d1)
        if a > b:
            a, b, r0, r1 = b, a, (r1 if r1 is not None else r0), r0
        return bm_cyl_x(a, b, r0, r1, seg=seg)

    body = [span(CUP_LEN, CUP_LEN - CUP_SOCKET_DEPTH - 0.0, CUP_STEM_OD / 2),
            span(CUP_LEN - CUP_SOCKET_DEPTH + 0.5, 0.0, CUP_STEM_OD / 2, CUP_RIM_OD / 2)]
    # cavity: cone from the floor to just past the mouth; socket for the hub spigot at the back
    slope = (CUP_MOUTH_D / 2 - CUP_CAVITY_FLOOR_D / 2) / CUP_CAVITY_DEPTH
    cavity = span(CUP_CAVITY_DEPTH, -1.0, CUP_CAVITY_FLOOR_D / 2, CUP_MOUTH_D / 2 + slope)
    socket = span(CUP_LEN + 1.0, CUP_LEN - CUP_SOCKET_DEPTH, CUP_SOCKET_D / 2)
    notches = []
    for k in range(4):
        n = bm_box(min(x(-1.0), x(CUP_NOTCH_DEPTH)), max(x(-1.0), x(CUP_NOTCH_DEPTH)),
                   -CUP_NOTCH_W / 2, CUP_NOTCH_W / 2, AXIS_Z + CUP_MOUTH_D / 2 - 3.0, AXIS_Z + CUP_RIM_OD / 2 + 2.0)
        notches.append(bm_rot_axis(n, 90.0 * k))
    ob = csg_solid(name, body, [cavity, socket] + notches, "SPINDLE", M("TPU_BLACK"), rotating=True)
    smooth(ob)
    return ob


def _label(k, rot):
    cu = bpy.data.curves.new(f"CUP_LABEL_{k}", "FONT")
    cu.body = str(k)
    cu.size = CUP_LABEL_H * MM
    cu.align_x = cu.align_y = "CENTER"
    cu.extrude = 0.0001
    ob = bpy.data.objects.new(f"CUP_LABEL_{k}", cu)
    coll("SPINDLE").objects.link(ob)
    ob.data.materials.append(M("LABEL_WHITE"))
    x_l = HEAD_MOUTH_X - 9.0
    r_outer = CUP_STEM_OD / 2 + (x_l - (HEAD_MOUTH_X - CUP_LEN + CUP_SOCKET_DEPTH - 0.5)) / (CUP_LEN - CUP_SOCKET_DEPTH + 0.5) * (CUP_RIM_OD - CUP_STEM_OD) / 2
    alpha = math.atan(((CUP_RIM_OD - CUP_STEM_OD) / 2) / (CUP_LEN - CUP_SOCKET_DEPTH + 0.5))
    pivot = Matrix.Translation(V(0, AXIS_Y, AXIS_Z))
    m = (pivot @ Matrix.Rotation(math.radians(90.0 * (k - 1)), 4, "X") @ Matrix.Translation(V(x_l, 0, r_outer + 0.15))
         @ Matrix.Rotation(-alpha, 4, "Y"))
    ob.matrix_world = m
    ob["label_for_notch"] = k
    ob["rotating"] = True
    parent_keep(ob, rot)
    return ob


def build_mango(rot):
    from mathutils import noise
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.0)
    a, b, c = MANGO_SEMI
    for v in bm.verts:
        nx, ny, nz = v.co.x, v.co.y, v.co.z
        taper = 1.0 - 0.09 * nx
        wob = 1.0 + 0.03 * noise.noise(Vector((2.2 * nx + 3.1, 2.2 * ny, 2.2 * nz)))
        px = a * nx
        edge = max(0.0, (abs(nx) - 0.80) / 0.20)
        px += math.copysign(MANGO_NUB * edge * edge, nx)
        py = b * ny * taper * wob + 3.5 * (1.0 - nx * nx)
        pz = c * nz * taper * wob
        v.co = Vector((px, py, pz)) * MM
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = new_obj("MANGO", bm, "MANGO", M("MANGO"), rotating=True, los_ignore=True)
    ob.location = V(MANGO_CX, AXIS_Y, AXIS_Z)
    for p in ob.data.polygons:
        p.use_smooth = True
    parent_keep(ob, rot)
    return ob


def build_spindle():
    rot = bpy.data.objects.new("SPINDLE_ROT", None)
    rot.empty_display_type = "ARROWS"
    rot.empty_display_size = 0.03
    rot.location = V(0, AXIS_Y, AXIS_Z)
    coll("SPINDLE").objects.link(rot)
    rot["scene_aux"] = True
    bpy.context.view_layer.update()
    petg = M("PETG_BLACK")
    # ---- motor (purchased) ----
    xb0 = MOTOR_FACE_X - MOTOR_LEN
    h = MOTOR_BODY / 2
    motor = bm_join(bm_box(xb0, MOTOR_FACE_X, AXIS_Y - h, AXIS_Y + h, AXIS_Z - h, AXIS_Z + h),
                    bm_cyl_x(MOTOR_FACE_X, MOTOR_FACE_X + MOTOR_BOSS_L, MOTOR_BOSS_D / 2, seg=40))
    new_obj("MOTOR_NEMA17", motor, "SPINDLE", M("MOTOR_GREY"))
    shaft = new_obj("MOTOR_SHAFT", bm_cyl_x(MOTOR_FACE_X, MOTOR_FACE_X + MOTOR_SHAFT_L, MOTOR_SHAFT_D / 2, seg=24),
                    "SPINDLE", M("STEEL"), rotating=True)
    parent_keep(shaft, rot)
    # ---- printed L-bracket with 12 o'clock pointer ----
    px0, px1 = MOTOR_FACE_X, MOTOR_FACE_X + BRACKET_PLATE_T
    hw = BRACKET_W / 2
    adds = [bm_box(px0, px1, -hw, hw, 0, BRACKET_TOP_Z),
            bm_box(BRACKET_FOOT_X0, BRACKET_FOOT_X1, -BRACKET_FOOT_W / 2, BRACKET_FOOT_W / 2, 0, BRACKET_FOOT_T),
            bm_box(px1 - 1, POINTER_X1, -POINTER_W / 2, POINTER_W / 2, BRACKET_TOP_Z - POINTER_T, BRACKET_TOP_Z),
            bm_box(POINTER_X1 - 2, POINTER_X1, -POINTER_W / 2, POINTER_W / 2, AXIS_Z + CUP_RIM_OD / 2 + 1.5, BRACKET_TOP_Z)]
    for s in (-1, 1):
        y0, y1 = sorted((s * (hw - 4.0), s * (hw - 0.5)))
        adds.append(bm_prism("xz", [(px0 + 0.5, 4), (px0 + 0.5, 75), (BRACKET_FOOT_X0 + 1.0, 4)], y0, y1))
        adds.append(bm_prism("xz", [(px1 - 0.5, 4), (px1 - 0.5, 75), (px1 + 22, 4)], y0, y1))
    cuts = [bm_cyl_x(px0 - 2, px1 + 2, BRACKET_BOSS_HOLE_D / 2, seg=48)]
    for sy in (-1, 1):
        for sz in (-1, 1):
            cuts.append(bm_cyl_x(px0 - 2, px1 + 2, BRACKET_M3_D / 2, y=sy * MOTOR_HOLE_PITCH / 2, z=AXIS_Z + sz * MOTOR_HOLE_PITCH / 2, seg=16))
    for fx in (BRACKET_FOOT_X0 + 6, BRACKET_FOOT_X1 - 6):
        for sy in (-1, 1):
            cuts.append(bm_cyl_z(fx, sy * (BRACKET_FOOT_W / 2 - 3.5), -1, BRACKET_FOOT_T + 1, FLOOR_SCREW_D / 2, 16))
    br = csg_solid("MOTOR_BRACKET", adds, cuts, "SPINDLE", petg)
    printed(br, (0, 0, 0), material="Black PETG", infill="40%", note="M3 x 4 into the motor, 4x M4 wood screws into the floor")
    # motor mounting screws (visual)
    heads = bm_join(*[bm_cyl_x(px1, px1 + 2.5, 2.75, y=sy * MOTOR_HOLE_PITCH / 2, z=AXIS_Z + sz * MOTOR_HOLE_PITCH / 2, seg=12)
                      for sy in (-1, 1) for sz in (-1, 1)])
    new_obj("MOTOR_SCREWS", heads, "SPINDLE", M("STEEL"))
    # ---- shaft hub ----
    hub = csg_solid("SHAFT_HUB",
                    [bm_cyl_x(HUB_X0, HUB_X1, HUB_D / 2, seg=40), bm_cyl_x(HUB_X1 - 0.5, HUB_X1 + SPIGOT_L, SPIGOT_D / 2, seg=40)],
                    [bm_cyl_x(HUB_X0 - 1, HUB_X0 + HUB_BORE_DEPTH, HUB_BORE_D / 2, seg=24),
                     bm_cyl_y(HUB_X0 + 7, 0, HUB_D, AXIS_Z, GRUB_D / 2, 16)],
                    "SPINDLE", petg, rotating=True)
    printed(hub, (0, -90, 0), material="Black PETG", infill="60%", note="M3 grub screw, M3 nut trap not modelled")
    parent_keep(hub, rot)
    # ---- cups ----
    head = _cup("CUP_HEAD", HEAD_MOUTH_X, -1)
    printed(head, (0, -90, 0), material="TPU 95A black", infill="20%", note="mouth up, no supports")
    parent_keep(head, rot)
    tail = _cup("CUP_TAIL", TAIL_MOUTH_NOMINAL_X, +1)
    printed(tail, (0, 90, 0), material="TPU 95A black", infill="20%", note="mouth up, no supports")
    parent_keep(tail, rot)
    for k in range(1, 5):
        _label(k, rot)
    build_mango(rot)
    build_tailstock(rot)


def build_tailstock(rot):
    petg = M("PETG_BLACK")
    steel = M("STEEL")
    # ---- rods (black oxide) ----
    for nm, s in (("ROD_L", -1), ("ROD_R", 1)):
        new_obj(nm, bm_cyl_x(ROD_X0, ROD_X1, ROD_D / 2, y=s * ROD_Y, z=ROD_Z, seg=24), "SPINDLE", M("BLACK_OXIDE"))
    # ---- rod holders (printed, one at each end) ----
    def holder(name, x0, x1, foot_x0, foot_x1, screw_x):
        adds = [bm_box(x0, x1, -HOLDER_Y, HOLDER_Y, 0, HOLDER_Z),
                bm_box(foot_x0, foot_x1, -HOLDER_Y, HOLDER_Y, 0, HOLDER_FOOT_T)]
        cuts = [bm_cyl_x(x0 - 1, x1 + 1, ROD_HOLE_D / 2, y=s * ROD_Y, z=ROD_Z, seg=24) for s in (-1, 1)]
        cuts += [bm_cyl_z(screw_x, s * 28.0, -1, HOLDER_FOOT_T + 1, FLOOR_SCREW_D / 2, 16) for s in (-1, 1)]
        ob = csg_solid(name, adds, cuts, "SPINDLE", petg)
        printed(ob, (0, 0, 0), material="Black PETG", infill="50%", note="rods retained with a drop of CA glue or an M3 grub screw")
        return ob
    holder("ROD_HOLDER_L", HOLDER_L_X[0], HOLDER_L_X[1], HOLDER_L_X[0], HOLDER_FOOT_L_X1, (HOLDER_L_X[1] + HOLDER_FOOT_L_X1) / 2)
    holder("ROD_HOLDER_R", HOLDER_R_X[0], HOLDER_R_X[1], HOLDER_FOOT_R_X0, HOLDER_R_X[1], (HOLDER_FOOT_R_X0 + HOLDER_R_X[0]) / 2)
    # ---- carriage (printed): 2x LM8UU, vertical rib, 608 bearing housing, pull-knob post ----
    bx = CARRIAGE_BX
    x0, x1 = HOUSING_X
    adds = [bm_box(bx - CARRIAGE_BLOCK_HALF_X, bx + CARRIAGE_BLOCK_HALF_X, -CARRIAGE_BLOCK_Y, CARRIAGE_BLOCK_Y,
                   CARRIAGE_BLOCK_Z[0], CARRIAGE_BLOCK_Z[1]),
            bm_box(x0, x1, -CARRIAGE_RIB_HALF_Y, CARRIAGE_RIB_HALF_Y, CARRIAGE_BLOCK_Z[1] - 1, AXIS_Z),
            bm_cyl_x(x0, x1, HOUSING_D / 2, seg=48),
            bm_prism("xz", [(x1 - 0.5, CARRIAGE_BLOCK_Z[1] - 0.5), (x1 - 0.5, AXIS_Z - 8), (x1 + CARRIAGE_GUSSET_LEN, CARRIAGE_BLOCK_Z[1] - 0.5)], -3, 3),
            bm_cyl_z(bx, KNOB_POST_Y, CARRIAGE_BLOCK_Z[1] - 1, KNOB_POST_TOP_Z, 5.0, 24)]
    cuts = [bm_cyl_x(bx - 16, bx + 16, BUSHING_POCKET_D / 2, y=s * ROD_Y, z=ROD_Z, seg=32) for s in (-1, 1)]
    cuts += [bm_cyl_x(x0 + HOUSING_WALL, x1 + 1, BEARING_RECESS_D / 2, seg=48),
             bm_cyl_x(x0 - 1, x0 + HOUSING_WALL + 0.5, HOUSING_FRONT_HOLE_D / 2, seg=40),
             bm_cyl_z(bx, KNOB_POST_Y, KNOB_POST_TOP_Z - 12, KNOB_POST_TOP_Z + 1, 2.4, 16)]
    car = csg_solid("CARRIAGE", adds, cuts, "SPINDLE", petg)
    printed(car, (0, 0, 0), material="Black PETG", infill="60%", note="press 608 from the +X side, secure with a drop of CA glue")
    # ---- pull knob (printed, glued onto the post) ----
    knob = csg_solid("PULL_KNOB",
                     [bm_cyl("z", (bx, KNOB_POST_Y, KNOB_POST_TOP_Z + KNOB_H / 2), 11.0, KNOB_H, 40, 14.0),
                      bm_cyl_z(bx, KNOB_POST_Y, KNOB_POST_TOP_Z - 11.0, KNOB_POST_TOP_Z + 1, 2.2, 16)],
                     [], "SPINDLE", petg)
    printed(knob, (180, 0, 0), material="Black PETG", infill="25%", note="print upside down, glue the stud into the post")
    # ---- purchased parts that ride with the carriage ----
    for nm, s in (("LM8UU_L", -1), ("LM8UU_R", 1)):
        new_obj(nm, bm_tube("x", (bx, s * ROD_Y, ROD_Z), LM8UU_OD / 2, ROD_D / 2, LM8UU_L, 32), "SPINDLE", steel)
    new_obj("BEARING_608", bm_tube("x", (x0 + HOUSING_WALL + BEARING_W / 2, AXIS_Y, AXIS_Z), BEARING_OD / 2, BEARING_ID / 2, BEARING_W, 40),
            "SPINDLE", steel)
    spring_x0 = bx + LM8UU_L / 2
    new_obj("SPRING", bm_helix_tube("x", spring_x0, HOLDER_R_X[0] - spring_x0, SPRING_OD / 2 - SPRING_WIRE_D / 2,
                                    SPRING_WIRE_D, SPRING_TURNS, (0, ROD_Y, ROD_Z)), "SPINDLE", steel)
    # ---- tail hub, bolt, nut (rotating with the fruit) ----
    f0, f1 = TAIL_HUB_FLANGE
    tail_mouth_back = TAIL_MOUTH_NOMINAL_X + CUP_LEN
    hub = csg_solid("TAIL_HUB",
                    [bm_cyl_x(tail_mouth_back - SPIGOT_L, f0 + 0.5, SPIGOT_D / 2, seg=40),
                     bm_cyl_x(f0, f1, TAIL_HUB_FLANGE_D / 2, seg=40),
                     bm_cyl_x(f1 - 0.5, TAIL_HUB_SPACER_X1, TAIL_HUB_SPACER_D / 2, seg=32)],
                    [bm_cyl_x(tail_mouth_back - SPIGOT_L - 1, TAIL_HUB_SPACER_X1 + 1, (BOLT_D + 0.4) / 2, seg=24),
                     bm_hex_x(tail_mouth_back - SPIGOT_L - 1, BOLT_HEAD[1], BOLT_HEAD_AF + 0.6)],
                    "SPINDLE", petg, rotating=True)
    printed(hub, (0, 90, 0), material="Black PETG", infill="60%", note="hex pocket takes the M8 bolt head")
    parent_keep(hub, rot)
    bolt = bm_join(bm_hex_x(BOLT_HEAD[0], BOLT_HEAD[1], BOLT_HEAD_AF), bm_cyl_x(BOLT_HEAD[1] - 0.2, BOLT_END_X, BOLT_D / 2, seg=24))
    b = new_obj("BOLT_M8", bolt, "SPINDLE", steel, rotating=True)
    parent_keep(b, rot)
    n = new_obj("NUT_M8", bm_hex_x(NUT[0], NUT[1], BOLT_HEAD_AF), "SPINDLE", steel, rotating=True)
    parent_keep(n, rot)


# =============================================================================
# CAMERAS: printed tower, adjustable carrier, boards and the two Blender cameras
# =============================================================================
def build_cameras():
    petg = M("PETG_BLACK")
    top_z = INT_Z + MDF_T
    fx0, fx1 = TOWER_FLANGE_X
    cx0, cx1 = TOWER_CHEEK_X
    ya, yb = TOWER_CHEEK_Y
    adds = [bm_box(fx0, fx1, -TOWER_FLANGE_Y, TOWER_FLANGE_Y, top_z, top_z + TOWER_FLANGE_T),
            bm_box(cx0, cx1, -yb, -ya, TOWER_BOTTOM_Z, top_z + 1),
            bm_box(cx0, cx1, ya, yb, TOWER_BOTTOM_Z, top_z + 1)]
    cuts = [bm_box(TOWER_OPEN_X[0], TOWER_OPEN_X[1], -TOWER_OPEN_Y, TOWER_OPEN_Y, top_z - 1, top_z + TOWER_FLANGE_T + 1)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            cuts.append(bm_cyl_z((fx0 + fx1) / 2 + sx * 50.0, sy * 33.0, top_z - 1, top_z + TOWER_FLANGE_T + 1, 2.15, 16))
    half = TOWER_SLOT_TRAVEL / 2
    for sx in TOWER_SLOT_XS:
        cuts.append(bm_box(sx - TOWER_SLOT_W / 2, sx + TOWER_SLOT_W / 2, ya - 1, yb + 1, TOWER_SLOT_ZC - half, TOWER_SLOT_ZC + half))
        for dz in (-half, half):
            cuts.append(bm_cyl_y(sx, ya - 1, yb + 1, TOWER_SLOT_ZC + dz, TOWER_SLOT_W / 2, 16))
    tower = csg_solid("CAMERA_TOWER", adds, cuts, "CAMERAS", petg, envelope_exempt=True, outside_frame=True)
    printed(tower, (180, 0, 0), material="Black PETG", infill="40%", note="flange down; 4x M4 wood screws; 2 slots give +-10 mm lens height")
    # carrier: horizontal plate that holds both boards + a tab bolted through the slots
    z0, z1 = CARRIER_PLATE_Z
    adds = [bm_box(CARRIER_PLATE_X[0], CARRIER_PLATE_X[1], CARRIER_PLATE_Y[0], CARRIER_PLATE_Y[1], z0, z1),
            bm_box(CARRIER_TAB_X[0], CARRIER_TAB_X[1], CARRIER_TAB_Y[0], CARRIER_TAB_Y[1], CARRIER_TAB_Z0, z0 + 1)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            adds.append(bm_cyl_z(CAM_RGB_X + sx * 15.0, sy * 15.0, z1 - 0.5, ARDUCAM_BOARD_Z0, 2.5, 16))
    cuts = [bm_cyl_z(CAM_NIR_X, CAM_Y, z0 - 1, z1 + 1, NOIR_BORE_D / 2, 32),
            bm_cyl_z(CAM_RGB_X, CAM_Y, z0 - 1, z1 + 1, ARDUCAM_BORE_D / 2, 40)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            cuts.append(bm_cyl_z(CAM_NIR_X + sx * NOIR_HOLES[0] / 2, sy * NOIR_HOLES[1] / 2, z0 - 1, z1 + 1, 1.1, 12))
            cuts.append(bm_cyl_z(CAM_RGB_X + sx * 15.0, sy * 15.0, z1, ARDUCAM_BOARD_Z0 + 1, 1.0, 12))
    for sx in TOWER_SLOT_XS:
        cuts.append(bm_cyl_y(sx, CARRIER_TAB_Y[0] - 1, CARRIER_TAB_Y[1] + 1, TOWER_SLOT_ZC, 1.65, 16))
    carrier = csg_solid("CAMERA_CARRIER", adds, cuts, "CAMERAS", petg, outside_frame=True)
    printed(carrier, (0, 0, 0), material="Black PETG", infill="40%", note="tree supports under the plate; tap M4 or use heat-set inserts in the tab")
    # purchased boards and lens barrels (children of the carrier so they follow height adjustments)
    parts = []
    bw, bh, bt = NOIR_BOARD
    parts.append(new_obj("NOIR_BOARD", bm_box(CAM_NIR_X - bw / 2, CAM_NIR_X + bw / 2, -bh / 2, bh / 2, z0 - bt, z0), "CAMERAS", M("PCB_GREEN")))
    parts.append(new_obj("NOIR_LENS", bm_cyl_z(CAM_NIR_X, CAM_Y, CAM_LENS_Z, z0 - bt + 0.2, NOIR_BARREL_D / 2, 24), "CAMERAS", M("BLACK_OXIDE")))
    aw, ah, at = ARDUCAM_BOARD
    parts.append(new_obj("ARDUCAM_BOARD", bm_box(CAM_RGB_X - aw / 2, CAM_RGB_X + aw / 2, -ah / 2, ah / 2, ARDUCAM_BOARD_Z0, ARDUCAM_BOARD_Z0 + at), "CAMERAS", M("PCB_BLUE")))
    parts.append(new_obj("ARDUCAM_LENS", bm_cyl_z(CAM_RGB_X, CAM_Y, CAM_LENS_Z, ARDUCAM_BOARD_Z0 + 0.2, ARDUCAM_BARREL_D / 2, 32), "CAMERAS", M("BLACK_OXIDE")))
    bpy.context.view_layer.update()
    for p in parts:
        parent_keep(p, carrier)
    for name, x in (("CAM_NIR", CAM_NIR_X), ("CAM_RGB", CAM_RGB_X)):
        cd = bpy.data.cameras.new(name)
        cd.sensor_fit = "HORIZONTAL"
        cd.sensor_width = SENSOR_W_MM
        cd.lens = FOCAL_MM
        cd.clip_start, cd.clip_end = 0.004, 1.5
        cd.display_size = 0.02
        cam = bpy.data.objects.new(name, cd)
        coll("CAMERAS").objects.link(cam)
        cam.location = V(x, CAM_Y, CAM_LENS_Z)
        cam.rotation_euler = (0.0, 0.0, 0.0)   # looks straight down -Z, image +X = world +X
        bpy.context.view_layer.update()
        parent_keep(cam, carrier)
    bpy.context.scene.camera = bpy.data.objects["CAM_RGB"]


# =============================================================================
# LIGHTING: two printed bars (front/back), 850 nm rings and white panels
# =============================================================================
def _face_frame(s):
    t = math.radians(BAR_TILT_DEG)
    n = Vector((0.0, s * math.sin(t), math.cos(t)))        # LED normal: up and outward
    w = Vector((0.0, -s * math.cos(t), math.sin(t)))       # up-slope (towards the fin)
    x = w.cross(n)
    return n, w, Matrix((x, w, n)).transposed()


def _add_area_light(name, shape, size, size_y, energy, pos, n, color, kind):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = shape
    ld.size = size * MM
    if shape == "RECTANGLE":
        ld.size_y = size_y * MM
    ld.energy = energy
    ld.color = color
    ld.spread = math.radians(130.0)
    ob = bpy.data.objects.new(name, ld)
    coll("LIGHTING").objects.link(ob)
    ob.location = pos
    ob.rotation_euler = n.to_track_quat("-Z", "Y").to_euler()
    ob["led_kind"] = kind
    ob["peak_energy"] = energy
    return ob


def build_lighting():
    petg = M("PETG_BLACK")
    t = math.radians(BAR_TILT_DEG)
    half = BAR_FACE_W / 2
    nd, nz = math.sin(t), math.cos(t)
    for s, tag in ((-1, "FRONT"), (1, "BACK")):
        t2 = "F" if s < 0 else "B"
        A = (BAR_FACE_CY - half * nz, BAR_FACE_CZ + half * nd)
        B = (BAR_FACE_CY + half * nz, BAR_FACE_CZ - half * nd)
        Ap = (A[0] - nd * BAR_PLATE_T, A[1] - nz * BAR_PLATE_T)
        Bp = (B[0] - nd * BAR_PLATE_T, B[1] - nz * BAR_PLATE_T)
        A2 = (Ap[0] + 2 * nd, Ap[1] + 2 * nz)
        B2 = (Bp[0] + 2 * nd, Bp[1] + 2 * nz)

        def yz(pts):
            return [(s * d, z) for d, z in pts]

        def yrange(d0, d1):
            return sorted((s * d0, s * d1))

        adds = [bm_prism("yz", yz([A, B, Bp, Ap]), BAR_PLATE_X[0], BAR_PLATE_X[1]),
                bm_box(BAR_X0, BAR_X1, *yrange(*BAR_FOOT_D), 0, BAR_FOOT_T),
                bm_box(BAR_FIN_X[0], BAR_FIN_X[1], *yrange(BAR_FIN_D0, BAR_FIN_D0 + BAR_FIN_T), BAR_FIN_Z0, BAR_FIN_TOP_Z)]
        for rx in BAR_RIB_XS:
            adds.append(bm_prism("yz", yz([A2, B2, (B2[0], 1.0), (BAR_RIB_FLOOR_D, 1.0), (BAR_FIN_D0 + 0.2, BAR_FIN_Z0)]),
                                 rx - BAR_RIB_T / 2, rx + BAR_RIB_T / 2))
        cuts = [bm_cyl_z(sx, s * BAR_FACE_CY, -1, BAR_FOOT_T + 1, FLOOR_SCREW_D / 2, 16) for sx in BAR_SCREW_XS]
        bar = csg_solid(f"LIGHT_BAR_{tag}", adds, cuts, "LIGHTING", petg, outside_frame=True)
        printed(bar, (0, 0, 0), material="Black PETG", infill="25%",
                note="supports under the LED face; 2x M4 wood screws; fin blocks every direct LED line")
        n, w, m3 = _face_frame(s)
        centre = Vector((BAR_CX, s * BAR_FACE_CY, BAR_FACE_CZ))

        def frame_at(dx):
            """Matrix mapping board-local metres to world, origin on the LED face at world X offset dx (mm)."""
            return Matrix.Translation(V(*(centre + Vector((dx, 0, 0))))) @ m3.to_4x4()

        def emitters(kind, dx, pts_local, board_top):
            f = frame_at(dx)
            for (xl, yl) in pts_local:
                Rig.led_emitters.append({"kind": kind, "bar": t2, "pos": f @ V(xl, yl, board_top + LED_DOME_H * 0.7)})

        def dome_mesh(dx, pts_local, board_top):
            f = frame_at(dx)
            doms = [bm_cyl("z", (xl, yl, board_top + LED_DOME_H / 2), LED_DOME_D / 2, LED_DOME_H, 10, LED_DOME_D / 3) for xl, yl in pts_local]
            return bm_xform(bm_join(*doms), f)

        # 850 nm ring (48 LEDs on two concentric circles)
        ring_pts = []
        for ci, (count, rad) in enumerate(RING_LED_RINGS):
            for i in range(count):
                a = 2 * math.pi * (i + 0.5 * ci) / count
                ring_pts.append((rad * math.cos(a), rad * math.sin(a)))
        bd = new_obj(f"NIR_RING_{t2}", bm_xform(bm_cyl("z", (0, 0, RING_T / 2), RING_D / 2, RING_T, 48), frame_at(0.0)),
                     "LIGHTING", M("PCB_BLUE"), outside_frame=True, los_ignore=True)
        new_obj(f"NIR_LEDS_{t2}", dome_mesh(0.0, ring_pts, RING_T), "LIGHTING", M("LED_NIR"), outside_frame=True, los_ignore=True,
                led_kind="nir")
        emitters("nir", 0.0, ring_pts, RING_T)
        _add_area_light(f"NIR_LIGHT_{t2}", "DISK", RING_D, RING_D, NIR_PEAK_W, (centre + n * (RING_T + 2.5)) * MM, n,
                        (1.0, 0.08, 0.04), "nir")
        # two white 36-LED panels
        rows, cols = PANEL_GRID
        grid = [((i - (rows - 1) / 2) * 9.0, (j - (cols - 1) / 2) * 5.5) for i in range(rows) for j in range(cols)]
        for k, off in enumerate(PANEL_X_OFFSETS, start=1):
            dx = off
            f = frame_at(dx)
            new_obj(f"WHITE_PANEL_{t2}_{k}", bm_xform(bm_box(-PANEL_ALONG_X / 2, PANEL_ALONG_X / 2, -PANEL_ACROSS / 2, PANEL_ACROSS / 2, 0, PANEL_T), f),
                    "LIGHTING", M("PCB_GREEN"), outside_frame=True, los_ignore=True)
            new_obj(f"WHITE_LEDS_{t2}_{k}", dome_mesh(dx, grid, PANEL_T), "LIGHTING", M("LED_WHITE"), outside_frame=True,
                    los_ignore=True, led_kind="white")
            emitters("white", dx, grid, PANEL_T)
            _add_area_light(f"WHITE_LIGHT_{t2}_{k}", "RECTANGLE", PANEL_ALONG_X, PANEL_ACROSS, WHITE_PEAK_W,
                            (centre + Vector((dx, 0, 0)) + n * (PANEL_T + 2.5)) * MM, n, (1.0, 0.97, 0.9), "white")


# =============================================================================
# ELECTRONICS: deck, button panel, jack housing, door switch
# =============================================================================
def build_electronics():
    petg = M("PETG_BLACK")
    top_z = INT_Z + MDF_T
    x0, x1 = DECK_X
    y0, y1 = DECK_Y
    z0 = top_z
    floor_top = z0 + DECK_FLOOR_T
    # board layout in deck-local mm (u along X, v along Y): name, u0, v0, size, hole inset, material
    layout = [("PI_ZERO_2W", 6.0, 94.0, PI_SIZE, None, "PCB_GREEN"),
              ("RELAY_2CH", 80.0, 84.0, RELAY_SIZE, 3.5, "PCB_BLUE"),
              ("STEPPER_DRIVER", 6.0, 40.0, DRIVER_SIZE, 3.0, "PCB_RED"),
              ("BUCK_LM2596", 64.0, 50.0, BUCK_SIZE, 2.5, "PCB_BLUE"),
              ("TERMINAL_BLOCK", 140.0, 20.0, TERMINAL_SIZE, 3.0, "PCB_GREEN")]
    posts, holes_ = [], []
    Rig.deck_rects = []
    for name, u0, v0, size, inset, _m in layout:
        Rig.deck_rects.append((name, u0, v0, u0 + size[0], v0 + size[1]))
        if name == "PI_ZERO_2W":
            hx_, hy_ = PI_HOLES
            pts = [(u0 + 3.5, v0 + 3.5), (u0 + 3.5 + hx_, v0 + 3.5), (u0 + 3.5, v0 + 3.5 + hy_), (u0 + 3.5 + hx_, v0 + 3.5 + hy_)]
        else:
            pts = [(u0 + inset, v0 + inset), (u0 + size[0] - inset, v0 + inset), (u0 + inset, v0 + size[1] - inset),
                   (u0 + size[0] - inset, v0 + size[1] - inset)]
        for pu, pv in pts:
            posts.append(bm_cyl_z(x0 + pu, y0 + pv, floor_top - 0.5, floor_top + DECK_POST_H, DECK_POST_D / 2, 16))
            holes_.append(bm_cyl_z(x0 + pu, y0 + pv, floor_top + 1.0, floor_top + DECK_POST_H + 1, 1.1, 10))
    cuts = [bm_box(x0 + DECK_WALL, x1 - DECK_WALL, y0 + DECK_WALL, y1 - DECK_WALL, floor_top, z0 + DECK_H + 1),
            bm_box(-62.0, -2.0, y0 - 1, y0 + DECK_WALL + 1, floor_top + 4, z0 + DECK_H + 1),      # camera cables in
            bm_box(-110.0, -90.0, y1 - DECK_WALL - 1, y1 + 1, floor_top + 4, z0 + DECK_H + 1)]   # supply cables out
    for su in (75.0, 175.0):
        for sv in (8.0, 122.0):
            cuts.append(bm_cyl_z(x0 + su, y0 + sv, z0 - 1, floor_top + 1, 2.15, 16))
    deck = csg_solid("ELECTRONICS_DECK", [bm_box(x0, x1, y0, y1, z0, z0 + DECK_H)] + posts, cuts + holes_, "ELECTRONICS", petg,
                     envelope_exempt=True)
    printed(deck, (0, 0, 0), material="Black PETG", infill="20%",
            note="190x130x35 fits the 200 mm bed unsplit; 4x M4 screws into the top panel; standoffs take M2.5 self-tapping screws")
    # purchased boards (visual stand-ins, same footprints as the real modules)
    top_of_posts = floor_top + DECK_POST_H
    for name, u0, v0, size, _inset, mname in layout:
        bx0, by0 = x0 + u0, y0 + v0
        t_pcb = 1.6
        parts = [bm_box(bx0, bx0 + size[0], by0, by0 + size[1], top_of_posts, top_of_posts + t_pcb)]
        zt = top_of_posts + t_pcb
        if name == "PI_ZERO_2W":
            parts.append(bm_box(bx0 + 20, bx0 + 33, by0 + 8, by0 + 21, zt, zt + 1.5))        # SoC shield
            parts.append(bm_box(bx0 + 1, bx0 + 12, by0 + 25, by0 + 29.5, zt, zt + 3.0))      # mini HDMI
        elif name == "RELAY_2CH":
            parts += [bm_box(bx0 + 6, bx0 + 21, by0 + 18, by0 + 37, zt, zt + 15.5), bm_box(bx0 + 27, bx0 + 42, by0 + 18, by0 + 37, zt, zt + 15.5)]
        elif name == "STEPPER_DRIVER":
            parts += [bm_box(bx0 + 8, bx0 + 28, by0 + 10, by0 + 25, zt, zt + 9), bm_cyl_z(bx0 + 40, by0 + 30, zt, zt + 12, 4.0, 16)]
        elif name == "BUCK_LM2596":
            parts += [bm_cyl_z(bx0 + 32, by0 + 11, zt, zt + 12, 6.0, 20), bm_box(bx0 + 6, bx0 + 16, by0 + 5, by0 + 16, zt, zt + 8)]
        elif name == "TERMINAL_BLOCK":
            parts = [bm_box(bx0, bx0 + size[0], by0, by0 + size[1], top_of_posts, top_of_posts + 10)]
        new_obj(name, bm_join(*parts), "ELECTRONICS", M(mname), envelope_exempt=True)
    # start button + status LED panel on the top-front edge
    px0, px1 = BUTTON_PANEL_X
    py0, py1 = BUTTON_PANEL_Y
    bz = top_z + BUTTON_PANEL_H / 2
    bx_btn, bx_led = px0 + 14.0, px1 - 10.0
    cuts = [bm_cyl_y(bx_btn, py0 - 1, py0 + 14, bz, (BUTTON_D + 0.2) / 2, 32),
            bm_cyl_y(bx_led, py0 - 1, py0 + 12, bz, (LED_STATUS_D + 0.2) / 2, 20)]
    for sx in (px0 + 6.0, px1 - 6.0):
        cuts.append(bm_cyl_z(sx, py1 - 4.0, top_z - 1, top_z + BUTTON_PANEL_H + 1, 2.15, 16))
    panel = csg_solid("BUTTON_PANEL", [bm_box(px0, px1, py0, py1, top_z, top_z + BUTTON_PANEL_H)], cuts, "ELECTRONICS", petg,
                      envelope_exempt=True)
    printed(panel, (0, 0, 0), material="Black PETG", infill="30%", note="16 mm button and 5 mm LED face the operator; 2x M4 screws from the top")
    new_obj("START_BUTTON", bm_cyl_y(bx_btn, py0 - 3.0, py0 + 9.0, bz, BUTTON_D / 2, 32), "ELECTRONICS", M("BRASS"), envelope_exempt=True)
    led = new_obj("STATUS_LED", bm_cyl_y(bx_led, py0 - 1.5, py0 + 6.0, bz, LED_STATUS_D / 2, 20), "ELECTRONICS", M("LED_STATUS"),
                  envelope_exempt=True, outside_frame=False)
    led["status_led"] = True
    # DC barrel jack housing on the back wall (outside)
    jx0, jy, jz = JACK_X - JACK_HOUSING[0] / 2, INT_Y / 2 + MDF_T, JACK_Z
    jbox = bm_box(jx0, jx0 + JACK_HOUSING[0], jy, jy + JACK_HOUSING[1], jz - JACK_HOUSING[2] / 2, jz + JACK_HOUSING[2] / 2)
    cuts = [bm_cyl_y(JACK_X, jy + 4, jy + JACK_HOUSING[1] + 1, jz, 5.6, 28),
            bm_box(JACK_X - 7.5, JACK_X + 7.5, jy + 5, jy + 15, jz - JACK_HOUSING[2] / 2 - 1, jz + 5.5)]   # pocket open at the bottom
    for sx, sz in ((JACK_X - 10.0, jz + 4.0), (JACK_X + 10.0, jz - 4.0)):
        cuts.append(bm_cyl_y(sx, jy - 1, jy + JACK_HOUSING[1] + 1, sz, 2.15, 16))
        cuts.append(bm_cyl_y(sx, jy + JACK_HOUSING[1] - 4.0, jy + JACK_HOUSING[1] + 1, sz, 4.2, 20))
    jack = csg_solid("JACK_HOUSING", [jbox], cuts, "ELECTRONICS", petg, envelope_exempt=True)
    printed(jack, (-90, 0, 0), material="Black PETG", infill="30%", note="holds a panel-mount DC barrel jack; 2x M4 wood screws into the back wall")
    new_obj("DC_JACK", bm_join(bm_cyl_y(JACK_X, jy + 8.0, jy + JACK_HOUSING[1] - 2.0, jz, 5.4, 24),
                               bm_box(JACK_X - 7, JACK_X + 7, jy + 6.0, jy + 14.0, jz - 5, jz + 5)),
            "ELECTRONICS", M("HOUSING_GREY"), envelope_exempt=True)
    # door micro-switch (inside, top right of the door frame) on a small printed bracket
    sx_, sy_, sz_ = SWITCH_POS
    wall_x = INT_X / 2 - LINER_T
    bracket = csg_solid("SWITCH_BRACKET",
                        [bm_box(wall_x - 4, wall_x, sy_ - 2, sy_ + 24, sz_ - 15, sz_ + 15),
                         bm_cyl_x(wall_x - 8, wall_x - 3.5, 2.0, y=sy_ + 5.0, z=sz_, seg=12),
                         bm_cyl_x(wall_x - 8, wall_x - 3.5, 2.0, y=sy_ + 15.0, z=sz_, seg=12)],
                        [bm_cyl_x(wall_x - 9, wall_x - 3.8, 0.9, y=sy_ + 5.0, z=sz_, seg=10),
                         bm_cyl_x(wall_x - 9, wall_x - 3.8, 0.9, y=sy_ + 15.0, z=sz_, seg=10),
                         bm_cyl_x(wall_x - 5, wall_x + 1, 2.15, y=sy_ + 10.0, z=sz_ + 10, seg=12),
                         bm_cyl_x(wall_x - 5, wall_x + 1, 2.15, y=sy_ + 10.0, z=sz_ - 10, seg=12)],
                        "ELECTRONICS", petg)
    printed(bracket, (0, 90, 0), material="Black PETG", infill="40%", note="M2 screws hold the micro-switch; 2x M4 screws through the liner into the wall")
    sw = bm_join(bm_box(wall_x - 14.4, wall_x - 8.0, sy_, sy_ + 20, sz_ - 5.1, sz_ + 5.1),
                 bm_box(wall_x - 12.5, wall_x - 11.5, sy_ - 10.5, sy_, sz_ - 1.5, sz_ - 0.5))
    new_obj("DOOR_SWITCH", sw, "ELECTRONICS", M("HOUSING_GREY"))


# =============================================================================
# CALIBRATION JIG (hidden variant): clamps between the cups like a mango
# =============================================================================
def build_calibration_jig(rot):
    petg = M("PETG_BLACK")
    plugs = [bm_cyl_x(HEAD_MOUTH_X - JIG_PLUG_LEN, HEAD_MOUTH_X + 0.5, 11.4, 17.9, seg=40),
             bm_cyl_x(TAIL_MOUTH_NOMINAL_X - 0.5, TAIL_MOUTH_NOMINAL_X + JIG_PLUG_LEN, 17.9, 11.4, seg=40)]
    spine = bm_box(HEAD_MOUTH_X, TAIL_MOUTH_NOMINAL_X, -JIG_BAR_Y / 2, JIG_BAR_Y / 2, AXIS_Z - JIG_BAR_Z / 2, AXIS_Z + JIG_BAR_Z / 2)
    plate_bottom = JIG_PLATE_TOP_Z - JIG_PLATE[2]
    posts = [bm_box(px - 4, px + 4, py - 4, py + 4, AXIS_Z, plate_bottom) for px in JIG_POST_XS for py in JIG_POST_YS]
    cuts = [bm_cyl_z(px, py, plate_bottom - 10, plate_bottom + 1, 1.5, 12) for px in JIG_POST_XS for py in JIG_POST_YS]
    body = csg_solid("JIG_BODY", [spine] + plugs + posts, cuts, "CALIBRATION_JIG", petg, variant="jig", rotating=True)
    printed(body, (0, 0, 0), material="Black PETG", infill="40%", note="cone plugs seat in the cups; plate bolted with 4x M3")
    parent_keep(body, rot)
    pw, pd, pt = JIG_PLATE
    plate = new_obj("JIG_PLATE", bm_box(-pw / 2, pw / 2, -pd / 2, pd / 2, -pt / 2, pt / 2), "CALIBRATION_JIG",
                    [M("WHITE_PVC"), M("ARUCO_BOARD")], variant="jig", rotating=True)
    for poly in plate.data.polygons:
        poly.material_index = 1 if poly.normal.z > 0.9 else 0
    plate.location = V(MANGO_CX, 0, JIG_PLATE_TOP_Z - pt / 2)
    bpy.context.view_layer.update()
    # cut the two screw holes through the plate (printed part)
    _boolean(plate, [bm_cyl_z(px, py, plate_bottom - 1, JIG_PLATE_TOP_Z + 1, 1.7, 12)
                         for px in JIG_POST_XS for py in JIG_POST_YS], "DIFFERENCE")
    for poly in plate.data.polygons:
        poly.material_index = 1 if poly.normal.z > 0.9 else 0
    cleanup_mesh(plate)
    printed(plate, (0, 0, 0), material="White PETG (glue the printed ArUco sheet on top)", infill="100%")
    parent_keep(plate, rot)
    for ob in (body, plate):
        ob.hide_viewport = True
        ob.hide_render = True


# =============================================================================
# WORLD, AUXILIARY CAMERAS/LIGHTS (not part of the rig) AND THE CAPTURE-CYCLE ANIMATION
# =============================================================================
def build_world_and_aux():
    sc = bpy.context.scene
    w = bpy.data.worlds.new("RIG_WORLD")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.6, 0.62, 0.66, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    sc.world = w
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, FRAME_END
    sc.view_settings.view_transform = "AgX"

    def aux_cam(name, loc, target, lens=35.0, ortho=None):
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        cd.display_size = 0.04
        cd.clip_start, cd.clip_end = 0.01, 6.0
        if ortho:
            cd.type = "ORTHO"
            cd.ortho_scale = ortho * MM
        ob = bpy.data.objects.new(name, cd)
        sc.collection.objects.link(ob)
        ob.location = V(*loc)
        d = V(*target) - ob.location
        ob.rotation_euler = (math.radians(90), 0, 0) if abs(d.x) + abs(d.z) < 1e-9 else d.to_track_quat("-Z", "Y").to_euler()
        ob["scene_aux"] = True
        return ob

    aux_cam("CAM_OVERVIEW", (-470, -1000, 860), (-10, -20, 185), lens=38.0)
    aux_cam("CAM_CUTAWAY", (-30, -1200, 190), (-30, 0, 190), ortho=760.0)
    aux_cam("CAM_ANIM", (-330, 800, 380), (-25, 0, 160), lens=30.0)
    for name, loc, energy, size in (("STUDIO_KEY", (-500, -900, 900), 160.0, 900.0), ("STUDIO_FILL", (900, -500, 500), 60.0, 900.0)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape, ld.size, ld.energy = "SQUARE", size * MM, energy
        ob = bpy.data.objects.new(name, ld)
        sc.collection.objects.link(ob)
        ob.location = V(*loc)
        ob.rotation_euler = (V(-30, 0, 160) - ob.location).to_track_quat("-Z", "Y").to_euler()
        ob["scene_aux"] = True
        ob.hide_render = True
        ob.hide_viewport = True
    # animation-only bounce stand-in under the ceiling (keyed together with the LED pulses)
    bl = bpy.data.lights.new("BOUNCE_FILL", "AREA")
    bl.shape, bl.size, bl.size_y, bl.energy, bl.spread = "RECTANGLE", 360 * MM, 280 * MM, 0.0, math.radians(150)
    bo = bpy.data.objects.new("BOUNCE_FILL", bl)
    sc.collection.objects.link(bo)
    bo.location = V(-30, 0, INT_Z - LINER_T - 4)
    bo["scene_aux"] = True
    bo["peak_energy"] = BOUNCE_PEAK_W
    bo.hide_render = True
    bo.hide_viewport = True


def _keys(setter, data_path_owner, prop, series, interp):
    """Insert keyframes (frame, value) with the requested interpolation."""
    pref = bpy.context.preferences.edit
    old = pref.keyframe_new_interpolation_type
    pref.keyframe_new_interpolation_type = interp
    try:
        for frame, value in series:
            setter(value)
            data_path_owner.keyframe_insert(prop, frame=frame)
    finally:
        pref.keyframe_new_interpolation_type = old


def _pulse(series, f_on, f_off, peak):
    """Append a linear on/off pulse to ``series`` (list of (frame, value))."""
    series += [(f_on - 1, 0.0), (f_on, peak), (f_off, peak), (f_off + 1, 0.0)]


def capture_timeline():
    """Frames of every event of the cycle (also used to choose the still-render frames)."""
    ev = {"white": [], "nir": [], "blink_rgb": [], "blink_nir": [], "rot": [], "still_rgb": [], "still_nir": []}
    for k, f0 in enumerate(FACE_START):
        ev["white"].append((f0 + W_ON, f0 + W_OFF))
        ev["nir"].append((f0 + N_ON, f0 + N_OFF))
        ev["blink_rgb"].append((f0 + BLINK_RGB[0], f0 + BLINK_RGB[1]))
        ev["blink_nir"].append((f0 + BLINK_NIR[0], f0 + BLINK_NIR[1]))
        ev["rot"].append((f0 + ROT_SPAN[0], f0 + ROT_SPAN[1]))
        ev["still_rgb"].append(f0 + HOLD_RGB)
        ev["still_nir"].append(f0 + HOLD_NIR)
    return ev


def build_animation():
    """Keyframes: door, spindle rotation, LED emission and light power (kept in the .blend for scrubbing)."""
    ev = capture_timeline()
    door = bpy.data.objects["DOOR"]
    rot = bpy.data.objects["SPINDLE_ROT"]
    open_z = -math.radians(DOOR_OPEN_DEG)
    # door: open -> closed at the start, open again at the end
    door_series = [(DOOR_CLOSE[0], open_z), (DOOR_CLOSE[1], 0.0), (DOOR_OPEN_AT[0], 0.0), (DOOR_OPEN_AT[1], open_z)]
    _keys(lambda v: setattr(door, "rotation_euler", (0.0, 0.0, v)), door, "rotation_euler", door_series, "BEZIER")
    # spindle: -90 deg after every face (labels 1,2,3,4 reach the top in order)
    rot_series = [(1, 0.0)]
    for k, (a, b) in enumerate(ev["rot"]):
        rot_series += [(a, math.radians(SPIN_STEP_DEG * k)), (b, math.radians(SPIN_STEP_DEG * (k + 1)))]
    rot_series.append((FRAME_END, math.radians(SPIN_STEP_DEG * 4)))
    _keys(lambda v: setattr(rot, "rotation_euler", (v, 0.0, 0.0)), rot, "rotation_euler", rot_series, "BEZIER")
    # light channels
    white, nir, blink = [(1, 0.0)], [(1, 0.0)], [(1, 0.0)]
    for (a, b) in ev["white"]:
        _pulse(white, a, b, 1.0)
    for (a, b) in ev["nir"]:
        _pulse(nir, a, b, 1.0)
    for (a, b) in ev["blink_rgb"] + ev["blink_nir"]:
        _pulse(blink, a, b, 1.0)
    blink.sort()
    white.append((FRAME_END, 0.0))
    nir.append((FRAME_END, 0.0))
    blink.append((FRAME_END, 0.0))
    for ob in bpy.data.objects:
        if ob.type == "LIGHT" and ob.get("led_kind") in ("white", "nir"):
            series = white if ob["led_kind"] == "white" else nir
            ld = ob.data
            peak = ob["peak_energy"]
            _keys(lambda v, ld=ld, peak=peak: setattr(ld, "energy", v * peak), ld, "energy", series, "LINEAR")
    fill = bpy.data.objects["BOUNCE_FILL"].data
    # union of the two pulse trains: either modality lights the box
    merged = {}
    for f, v in white + nir:
        merged[f] = max(merged.get(f, 0.0), v)
    _keys(lambda v: setattr(fill, "energy", v * BOUNCE_PEAK_W), fill, "energy", sorted(merged.items()), "LINEAR")
    for mname, series in (("LED_WHITE", white), ("LED_NIR", nir), ("LED_STATUS", blink)):
        sock = M(mname).node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
        peak = EMISSION_PEAK[mname]
        _keys(lambda v, sock=sock, peak=peak: setattr(sock, "default_value", v * peak), sock, "default_value", series, "LINEAR")
    sc = bpy.context.scene
    sc.frame_set(1)
    bpy.context.view_layer.update()


# <<BUILDERS>>

# =============================================================================
# CHECKS (written first; see TDD notes in the README)
# =============================================================================
def _objs(**props):
    """Mesh/light/camera objects whose custom properties match ``props``."""
    out = []
    for ob in bpy.data.objects:
        if all(ob.get(k) == v for k, v in props.items()):
            out.append(ob)
    return out


def _mesh_objs():
    return [o for o in bpy.data.objects if o.type == "MESH" and not o.get("scene_aux")]


def _world_verts(ob):
    mw = ob.matrix_world
    return np.array([tuple(mw @ v.co) for v in ob.data.vertices], dtype=float).reshape(-1, 3)


def _world_bbox_mm(ob):
    if ob.type == "MESH" and len(ob.data.vertices):
        pts = _world_verts(ob) / MM
        return pts.min(axis=0), pts.max(axis=0)
    p = np.array(tuple(ob.matrix_world.translation)) / MM
    return p, p


def _surface_samples(ob, spacing_mm=10.0, max_div=40):
    """World-space points covering every triangle of ``ob`` (metres)."""
    me = ob.data
    me.calc_loop_triangles()
    mw = ob.matrix_world
    verts = np.array([tuple(mw @ v.co) for v in me.vertices], dtype=float).reshape(-1, 3)
    pts = [verts]
    for tri in me.loop_triangles:
        a, b, c = (verts[i] for i in tri.vertices)
        longest = max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)) / MM
        k = int(min(max_div, max(1, math.ceil(longest / spacing_mm))))
        if k == 1:
            pts.append(((a + b + c) / 3.0)[None, :])
            continue
        ij = [(i, j) for i in range(k + 1) for j in range(k + 1 - i)]
        w = np.array([(i / k, j / k, 1 - i / k - j / k) for i, j in ij])
        pts.append(w @ np.stack([a, b, c]))
    return np.vstack(pts)


def _camera_tans(cam):
    """Half-angle tangents (x, y) of a Blender camera for the scene resolution."""
    cd = cam.data
    sc = bpy.context.scene
    aspect = sc.render.resolution_y / sc.render.resolution_x
    half_w = (cd.sensor_width / 2.0) / cd.lens
    return half_w, half_w * aspect


def _in_frustum(points, cam, margin=0.0):
    """Boolean mask of ``points`` (N,3 metres) inside ``cam``'s view pyramid (looking down -Z)."""
    tx, ty = _camera_tans(cam)
    o = np.array(tuple(cam.matrix_world.translation))
    d = o[2] - points[:, 2]
    inside = (d > 0)
    inside &= np.abs(points[:, 0] - o[0]) < (tx * d - margin)
    inside &= np.abs(points[:, 1] - o[1]) < (ty * d - margin)
    return inside


def _build_bvh(objs):
    verts, polys = [], []
    for ob in objs:
        me = ob.data
        me.calc_loop_triangles()
        base = len(verts)
        mw = ob.matrix_world
        verts.extend(tuple(mw @ v.co) for v in me.vertices)
        polys.extend(tuple(base + i for i in t.vertices) for t in me.loop_triangles)
    if not polys:
        return None
    return BVHTree.FromPolygons(verts, polys, epsilon=0.0)


def check_a_frustum_footprint():
    """(a) both camera frustums contain the 175x110 mm fruit footprint at Z=150."""
    fails = []
    corners = np.array([[FOOTPRINT_CX + sx * FOOTPRINT_X / 2, sy * FOOTPRINT_Y / 2, CAL_PLANE_Z]
                        for sx in (-1, 1) for sy in (-1, 1)]) * MM
    for name in ("CAM_NIR", "CAM_RGB"):
        cam = bpy.data.objects.get(name)
        if cam is None or cam.type != "CAMERA":
            fails.append(f"(a) {name} is missing")
            continue
        if not _in_frustum(corners, cam).all():
            fails.append(f"(a) {name} frustum does not contain the {FOOTPRINT_X:g}x{FOOTPRINT_Y:g} mm footprint at Z={CAL_PLANE_Z:g}")
    return fails


def check_camera_setup():
    """Camera objects match the agreed optics, positions and baseline."""
    fails = []
    want = {"CAM_NIR": CAM_NIR_X, "CAM_RGB": CAM_RGB_X}
    for name, x in want.items():
        cam = bpy.data.objects.get(name)
        if cam is None or cam.type != "CAMERA":
            fails.append(f"(cam) {name} is missing")
            continue
        cd = cam.data
        loc = np.array(tuple(cam.matrix_world.translation)) / MM
        if np.abs(loc - np.array([x, CAM_Y, CAM_LENS_Z])).max() > 0.01:
            fails.append(f"(cam) {name} located at {loc.round(3)} instead of {(x, CAM_Y, CAM_LENS_Z)}")
        if cd.sensor_fit != "HORIZONTAL" or abs(cd.sensor_width - SENSOR_W_MM) > 1e-6 or abs(cd.lens - FOCAL_MM) > 1e-6:
            fails.append(f"(cam) {name} optics are not IMX219 (sensor_fit HORIZONTAL, 3.68 mm, 3.04 mm)")
    sc = bpy.context.scene
    if (sc.render.resolution_x, sc.render.resolution_y) != (RES_X, RES_Y):
        fails.append(f"(cam) scene resolution {(sc.render.resolution_x, sc.render.resolution_y)} != {(RES_X, RES_Y)}")
    return fails


def check_b_tailstock_travel():
    """(b) tailstock travel gives 80-175 mm and the carriage stays on the rods."""
    fails = []
    head = bpy.data.objects.get("CUP_HEAD")
    tail = bpy.data.objects.get("CUP_TAIL")
    car = bpy.data.objects.get("CARRIAGE")
    hl, hr = bpy.data.objects.get("ROD_HOLDER_L"), bpy.data.objects.get("ROD_HOLDER_R")
    rods = [bpy.data.objects.get("ROD_L"), bpy.data.objects.get("ROD_R")]
    if None in (head, tail, car, hl, hr) or None in rods:
        return ["(b) spindle/tailstock objects are missing"]
    head_mouth = _world_bbox_mm(head)[1][0]
    tail_mouth_nom = _world_bbox_mm(tail)[0][0]
    lo = (tail_mouth_nom + (TAIL_MOUTH_MIN_X - TAIL_MOUTH_NOMINAL_X)) - head_mouth
    hi = (tail_mouth_nom + (TAIL_MOUTH_MAX_X - TAIL_MOUTH_NOMINAL_X)) - head_mouth
    if abs(lo - FRUIT_LEN_MIN) > 0.01 or abs(hi - FRUIT_LEN_MAX) > 0.01:
        fails.append(f"(b) fruit length range is {lo:.1f}-{hi:.1f} mm, expected {FRUIT_LEN_MIN:g}-{FRUIT_LEN_MAX:g}")
    cmin, cmax = _world_bbox_mm(car)
    shift_lo = TAIL_MOUTH_MIN_X - tail_mouth_nom
    shift_hi = TAIL_MOUTH_MAX_X - tail_mouth_nom

    def body_face(ob, pick_max):
        pts = _world_verts(ob) / MM
        body = pts[pts[:, 2] > HOLDER_FOOT_T + 0.5]
        return body[:, 0].max() if pick_max else body[:, 0].min()

    left_face = body_face(hl, True)
    right_face = body_face(hr, False)
    bush = bpy.data.objects.get("LM8UU_R")
    bush_max = _world_bbox_mm(bush)[1][0] if bush is not None else cmax[0]
    spring_solid = SPRING_WIRE_D * (SPRING_TURNS + 2)
    if cmin[0] + shift_lo < left_face:
        fails.append(f"(b) carriage hits the left rod holder at minimum travel ({cmin[0] + shift_lo:.1f} < {left_face:.1f})")
    if cmax[0] + shift_hi > right_face:
        fails.append(f"(b) carriage hits the right rod holder at maximum travel ({cmax[0] + shift_hi:.1f} > {right_face:.1f})")
    if bush_max + shift_hi + spring_solid > right_face:
        fails.append(f"(b) spring would be solid at maximum travel ({bush_max + shift_hi + spring_solid:.1f} > {right_face:.1f})")
    for r in rods:
        rmin, rmax = _world_bbox_mm(r)
        if rmin[0] > ROD_X0 + 0.01 or rmax[0] < ROD_X1 - 0.01:
            fails.append(f"(b) {r.name} does not span X={ROD_X0:g}..{ROD_X1:g}")
    return fails


def check_c_rotation_clearance():
    """(c) a 55 mm-radius fruit rotating about the axis clears floor, rods, bars and fins."""
    segs = 96
    r = FRUIT_CLEARANCE_R * MM / math.cos(math.pi / segs)   # circumscribed polygon
    x0, x1 = HEAD_MOUTH_X * MM, TAIL_MOUTH_NOMINAL_X * MM
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=r, radius2=r, depth=x1 - x0,
                          matrix=Matrix.Translation(((x0 + x1) / 2, AXIS_Y * MM, AXIS_Z * MM)) @ Matrix.Rotation(math.pi / 2, 4, "Y"))
    bmesh.ops.triangulate(bm, faces=bm.faces)
    verts = [tuple(v.co) for v in bm.verts]
    polys = [tuple(v.index for v in f.verts) for f in bm.faces]
    bm.free()
    cyl = BVHTree.FromPolygons(verts, polys)
    fails = []
    statics = [o for o in _mesh_objs() if not o.get("rotating") and o.get("variant") != "jig" and o.get("clearance_ignore") is None]
    if not statics:
        return ["(c) no static objects to test against (scene is empty)"]
    required = {"FLOOR", "ROD_L", "ROD_R", "LIGHT_BAR_FRONT", "LIGHT_BAR_BACK"}
    missing = sorted(required - {o.name for o in statics})
    if missing:
        fails.append(f"(c) missing objects: {', '.join(missing)}")
    for ob in statics:
        pts = _world_verts(ob)
        if len(pts):
            rad = np.hypot(pts[:, 1] - AXIS_Y * MM, pts[:, 2] - AXIS_Z * MM)
            inside = (rad < FRUIT_CLEARANCE_R * MM) & (pts[:, 0] > x0) & (pts[:, 0] < x1)
            if inside.any():
                fails.append(f"(c) {ob.name} has vertices inside the rotating {FRUIT_CLEARANCE_R:g} mm fruit envelope")
                continue
        tree = _build_bvh([ob])
        if tree is not None and cyl.overlap(tree):
            fails.append(f"(c) {ob.name} intersects the rotating {FRUIT_CLEARANCE_R:g} mm fruit envelope")
    return fails


def check_d_print_volume():
    """(d) every printed part fits a 200x200x200 mm print volume."""
    fails = []
    parts = _objs(printed=True)
    if not parts:
        return ["(d) no printed parts found"]
    for ob in parts:
        lo, hi = _world_bbox_mm(ob)
        dims = hi - lo
        if (dims > PRINT_VOLUME + BBOX_TOL).any():
            fails.append(f"(d) {ob.name} is {dims.round(2)} mm, larger than {PRINT_VOLUME:g}^3")
    return fails


def check_e_manifold():
    """(e) every printed mesh is manifold (zero non-manifold edges)."""
    fails = []
    parts = _objs(printed=True)
    if not parts:
        return ["(e) no printed parts found"]
    for ob in parts:
        if ob.type != "MESH":
            fails.append(f"(e) printed object {ob.name} is not a mesh")
            continue
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bad_edges = [e for e in bm.edges if not e.is_manifold]
        loose = sum(1 for v in bm.verts if not v.link_edges)
        where = [tuple(round(c / MM, 1) for c in (ob.matrix_world @ ((e.verts[0].co + e.verts[1].co) / 2))) for e in bad_edges[:3]]
        bm.free()
        if bad_edges or loose:
            fails.append(f"(e) {ob.name}: {len(bad_edges)} non-manifold edges (e.g. at {where} mm), {loose} loose vertices")
    return fails


def check_f_no_direct_light():
    """(f) no straight line from any LED emitter to the fruit top or to a lens."""
    fails = []
    if not Rig.led_emitters:
        return ["(f) no LED emitters registered"]
    mango = bpy.data.objects.get("MANGO")
    if mango is None:
        return ["(f) MANGO is missing"]
    occluders = [o for o in _mesh_objs() if not o.get("los_ignore") and o.get("variant") != "jig"
                 and o.name != "MANGO" and not o.name.startswith(("LINER", "WALL", "FLOOR", "TOP_", "DOOR"))]
    tree = _build_bvh(occluders)
    if tree is None:
        return ["(f) no occluders in the scene"]
    top_z = (AXIS_Z + FRUIT_CLEARANCE_R) * MM
    targets = [("fruit-top", Vector((x * MM, 0.0, top_z))) for x in (-95, -64, -33, -2, 29)]
    for name in ("CAM_NIR", "CAM_RGB"):
        cam = bpy.data.objects.get(name)
        if cam is None:
            fails.append(f"(f) {name} is missing")
            continue
        # follow the camera so check (k) can test the whole slot travel
        targets.append((f"lens {name}", cam.matrix_world.translation - Vector((0.0, 0.0, 0.5 * MM))))
    # upper-hemisphere surface samples of the nominal fruit (what the cameras actually see)
    mw = mango.matrix_world
    me = mango.data
    for v in list(me.vertices)[:: max(1, len(me.vertices) // 60)]:
        p = mw @ v.co
        if p.z > AXIS_Z * MM + 0.01:
            targets.append(("fruit-surface", p + Vector((0, 0, 0.0005))))
    seen = {}
    for led in Rig.led_emitters:
        o = led["pos"]
        for label, t in targets:
            d = t - o
            dist = d.length
            hit = tree.ray_cast(o, d.normalized(), dist - 1e-4)
            if hit[0] is None:
                key = (led["kind"], led["bar"], label)
                seen[key] = seen.get(key, 0) + 1
    for (kind, bar, label), n in sorted(seen.items()):
        fails.append(f"(f) {n} {kind} LED(s) on bar {bar} have a direct line to {label}")
    return fails


def check_g_outside_frames():
    """(g) fins, light bars and tower skirts are outside both camera frames."""
    fails = []
    cams = [bpy.data.objects.get("CAM_NIR"), bpy.data.objects.get("CAM_RGB")]
    if None in cams:
        return ["(g) cameras are missing"]
    parts = _objs(outside_frame=True)
    if not parts:
        return ["(g) no objects tagged outside_frame"]
    for ob in parts:
        if ob.type != "MESH":
            continue
        pts = _surface_samples(ob)
        for cam in cams:
            n = int(_in_frustum(pts, cam).sum())
            if n:
                fails.append(f"(g) {ob.name} enters the {cam.name} frame ({n} sample points)")
    return fails


def check_h_envelope():
    """(h) everything lies inside the external envelope except deck, button, jack and the open door."""
    fails = []
    lo = np.array([-ENV_HALF_X, -ENV_HALF_Y, ENV_Z0])
    hi = np.array([ENV_HALF_X, ENV_HALF_Y, ENV_Z1])
    door = bpy.data.objects.get("DOOR")
    saved = None
    if door is not None:
        saved = door.rotation_euler.z
        door.rotation_euler.z = 0.0
        bpy.context.view_layer.update()
    try:
        candidates = _mesh_objs() + [o for o in bpy.data.objects if o.type in {"LIGHT", "CAMERA"} and not o.get("scene_aux")]
        if len(candidates) < 5:
            return ["(h) scene is empty"]
        for ob in candidates:
            if ob.get("envelope_exempt"):
                continue
            a, b = _world_bbox_mm(ob)
            if (a < lo - 1e-3).any() or (b > hi + 1e-3).any():
                fails.append(f"(h) {ob.name} sticks out of the envelope: {a.round(1)} .. {b.round(1)}")
    finally:
        if door is not None:
            door.rotation_euler.z = saved
            bpy.context.view_layer.update()
    return fails


def check_i_leds_and_layout():
    """LED counts and electronics-deck layout sanity."""
    fails = []
    n_nir = sum(1 for l in Rig.led_emitters if l["kind"] == "nir")
    n_white = sum(1 for l in Rig.led_emitters if l["kind"] == "white")
    if (n_nir, n_white) != (96, 144):
        fails.append(f"(i) LED emitters are nir={n_nir}, white={n_white}; expected 96 and 144")
    rects = Rig.deck_rects
    if not rects:
        fails.append("(i) electronics deck layout is empty")
    ix0, ix1 = 0.0, DECK_X[1] - DECK_X[0]
    iy0, iy1 = 0.0, DECK_Y[1] - DECK_Y[0]
    for i, (n, x0, y0, x1, y1) in enumerate(rects):
        if x0 < ix0 + DECK_WALL or x1 > ix1 - DECK_WALL or y0 < iy0 + DECK_WALL or y1 > iy1 - DECK_WALL:
            fails.append(f"(i) deck part {n} leaves the tray")
        for m, a0, b0, a1, b1 in rects[i + 1:]:
            if x0 < a1 and a0 < x1 and y0 < b1 and b0 < y1:
                fails.append(f"(i) deck parts {n} and {m} overlap")
    return fails


def check_j_calibration_jig():
    """Calibration plate top sits at Z=150 and both cameras see the whole plate."""
    fails = []
    plate = bpy.data.objects.get("JIG_PLATE")
    if plate is None:
        return ["(j) JIG_PLATE is missing"]
    lo, hi = _world_bbox_mm(plate)
    if abs(hi[2] - JIG_PLATE_TOP_Z) > 0.01:
        fails.append(f"(j) jig plate top is at Z={hi[2]:.2f}, expected {JIG_PLATE_TOP_Z:g}")
    corners = np.array([[x, y, hi[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) * MM
    for name in ("CAM_NIR", "CAM_RGB"):
        cam = bpy.data.objects.get(name)
        if cam is not None and not _in_frustum(corners, cam).all():
            fails.append(f"(j) {name} does not see the whole calibration plate")
    return fails


CAMERA_HEIGHT_CHECKS = (check_a_frustum_footprint, check_f_no_direct_light,
                        check_g_outside_frames, check_j_calibration_jig)


def check_k_camera_travel():
    """(k) the camera-height dependent checks hold at both ends of the tower slot travel."""
    carrier = bpy.data.objects.get("CAMERA_CARRIER")
    if carrier is None:
        return ["(k) CAMERA_CARRIER is missing"]
    fails = []
    z_nominal = carrier.location.z
    try:
        for dz in (-TOWER_SLOT_TRAVEL / 2, TOWER_SLOT_TRAVEL / 2):
            carrier.location.z = z_nominal + dz * MM   # boards, lenses and cameras are its children
            bpy.context.view_layer.update()
            for fn in CAMERA_HEIGHT_CHECKS:
                fails.extend(f"(k) lens at Z={CAM_LENS_Z + dz:g}: {msg}" for msg in fn())
    finally:
        carrier.location.z = z_nominal
        bpy.context.view_layer.update()
    return fails


ALL_CHECKS = (check_camera_setup, check_a_frustum_footprint, check_b_tailstock_travel,
              check_c_rotation_clearance, check_d_print_volume, check_e_manifold,
              check_f_no_direct_light, check_g_outside_frames, check_h_envelope,
              check_i_leds_and_layout, check_j_calibration_jig, check_k_camera_travel)


def run_checks():
    """Run every check; returns the list of failure strings."""
    sc = bpy.context.scene
    sc.frame_set(sc.frame_start)
    bpy.context.view_layer.update()
    failures = []
    for fn in ALL_CHECKS:
        try:
            res = fn()
        except Exception as exc:  # a crashing check is a failing check
            res = [f"({fn.__name__}) raised {type(exc).__name__}: {exc}"]
        print(f"[check] {fn.__name__}: {'ok' if not res else 'FAIL'}")
        failures.extend(res)
    return failures


# =============================================================================
# EXPORT: ArUco board, STLs, MDF cut list, .blend
# =============================================================================
def ensure_aruco_board():
    """Make sure the printable ArUco board exists (generated with OpenCV from the repo venv)."""
    target = Rig.root / ARUCO_FILE
    if target.exists():
        return True
    helper = Rig.root / "calibration" / "make_aruco_board.py"
    for py in (os.environ.get("CAPTURE_RIG_PYTHON"), REPO_VENV_PY):
        if py and Path(py).exists() and helper.exists():
            res = subprocess.run([py, "-I", str(helper)], capture_output=True, text=True)
            if res.returncode == 0 and target.exists():
                return True
    print("WARNING: ArUco board not generated (cv2.aruco unavailable); the jig plate uses a checker texture")
    return False


def _print_pose_points(ob):
    """World vertex positions (Vector, metres) of ``ob`` rotated into its print orientation, resting on Z=0 and centred in XY."""
    mw = ob.matrix_world
    rot = Rig.print_orient.get(ob.name, Euler()).to_matrix().to_4x4()
    pts = [mw @ v.co for v in ob.data.vertices]
    centre = Vector((sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts), sum(p.z for p in pts) / len(pts)))
    pts = [rot @ (p - centre) for p in pts]
    off = Vector((0.0, 0.0, 0.0))
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    off.x, off.y, off.z = -(max(xs) + min(xs)) / 2, -(max(ys) + min(ys)) / 2, -min(zs)
    return [p + off for p in pts]


def export_stls(stl_dir):
    stl_dir.mkdir(parents=True, exist_ok=True)
    for old in stl_dir.glob("*.stl"):
        old.unlink()
    sc = bpy.context.scene
    tmp = bpy.data.collections.new("TMP_STL")
    sc.collection.children.link(tmp)
    written = []
    for ob in sorted(_objs(printed=True), key=lambda o: o.name):
        me = ob.data.copy()
        flat = np.array([c for p in _print_pose_points(ob) for c in (p.x, p.y, p.z)], dtype=np.float32)
        me.vertices.foreach_set("co", flat)
        me.update()
        t = bpy.data.objects.new(ob.name, me)
        tmp.objects.link(t)
        bpy.ops.object.select_all(action="DESELECT")
        t.select_set(True)
        bpy.context.view_layer.objects.active = t
        path = stl_dir / f"{ob.name.lower()}.stl"
        bpy.ops.wm.stl_export(filepath=str(path), export_selected_objects=True, global_scale=1000.0,
                              ascii_format=False, apply_modifiers=False)
        bpy.data.objects.remove(t)
        bpy.data.meshes.remove(me)
        written.append(path)
    sc.collection.children.unlink(tmp)
    bpy.data.collections.remove(tmp)
    for path in written:  # read back: millimetres, resting on Z=0, inside the print volume
        raw = path.read_bytes()
        n = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
        rec = np.frombuffer(raw[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        pts = rec["v"].reshape(-1, 3)
        dims = pts.max(axis=0) - pts.min(axis=0)
        print(f"[stl] {path.name:26s} {n:6d} tris  {dims[0]:7.1f} x {dims[1]:7.1f} x {dims[2]:7.1f} mm  zmin={pts[:, 2].min():.2f}")
        if (dims > PRINT_VOLUME + BBOX_TOL).any():
            raise RuntimeError(f"{path.name} exceeds the print volume: {dims}")
    return written


PART_NOTES = {
    "FLOOR": "floor; paint the upper face matte black",
    "TOP_PANEL": "top panel with the camera opening ({opening}); paint the lower face matte black",
    "WALL_L": "side wall (stands on the floor, carries the top panel); paint the inner face matte black",
    "WALL_R": "side wall (stands on the floor, carries the top panel); paint the inner face matte black",
    "WALL_BACK": "back wall, {holes}; paint the inner face matte black",
    "DOOR": "front door hinged on the left (nominal size: cut 1 mm smaller per side for clearance); paint the inner face matte black",
    "LINER_L": "white lining, upper walls (lower edge at Z=130)",
    "LINER_R": "white lining, upper walls (lower edge at Z=130)",
    "LINER_BACK": "white lining, upper walls (lower edge at Z=130)",
    "LINER_DOOR": "white lining, upper door (lower edge at Z=130)",
    "LINER_CEILING": "white lining, ceiling, with the camera opening ({opening})",
}


def export_cut_list(path):
    groups = {}
    for ob in _objs(mdf=True) + _objs(pvc=True):
        pts = np.array([tuple(v.co) for v in ob.data.vertices]) / MM
        dims = tuple(sorted((pts.max(axis=0) - pts.min(axis=0)).round(1), reverse=True))
        kind = "MDF 9 mm" if ob.get("mdf") else "PVC foam board 3 mm"
        note = PART_NOTES.get(ob.name, "").format(opening=ob.get("opening", ""), holes=ob.get("holes", ""))
        groups.setdefault((kind, dims, note), []).append(ob.name)
    order = sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1][0] * kv[0][1][1], kv[0][2]))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "qty", "length_mm", "width_mm", "thickness_mm", "material", "used_for", "notes"])
        for i, ((kind, dims, note), names) in enumerate(order, start=1):
            w.writerow([f"{'M' if 'MDF' in kind else 'P'}{i}", len(names), f"{dims[0]:g}", f"{dims[1]:g}", f"{dims[2]:g}", kind,
                        "+".join(sorted(names)), note])
    print(f"[csv] {path.name}: {len(order)} cut sizes, {sum(len(n) for _, n in order)} pieces")


def save_blend(path):
    sc = bpy.context.scene
    sc.frame_set(1)
    sc.camera = bpy.data.objects["CAM_OVERVIEW"]
    bpy.context.preferences.filepaths.save_version = 0   # no .blend1 backup next to the deliverable
    bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True)
    print(f"[blend] {path}")


# =============================================================================
# RENDERS: stills (Cycles + OIDN) and the capture-cycle animation (EEVEE)
# =============================================================================
class hidden:
    """Context manager: hide objects from render (by exact name or prefix) or force-show hidden ones."""

    def __init__(self, names=(), prefixes=(), show=()):
        self.names, self.prefixes, self.show, self.saved = set(names), tuple(prefixes), set(show), {}

    def __enter__(self):
        for ob in bpy.data.objects:
            if ob.name in self.show:
                self.saved[ob] = (ob.hide_render, ob.hide_viewport)
                ob.hide_render = ob.hide_viewport = False
            elif ob.name in self.names or (self.prefixes and ob.name.startswith(self.prefixes)):
                self.saved[ob] = (ob.hide_render, ob.hide_viewport)
                ob.hide_render = True
        return self

    def __exit__(self, *exc):
        for ob, (r, v) in self.saved.items():
            ob.hide_render, ob.hide_viewport = r, v


def set_world_strength(value):
    bpy.data.worlds["RIG_WORLD"].node_tree.nodes["Background"].inputs["Strength"].default_value = value


def configure_cycles(sc, samples):
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.02
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.max_bounces = 8
    cy.diffuse_bounces = 6
    cy.glossy_bounces = 3
    cy.transmission_bounces = 4
    cy.sample_clamp_indirect = 6.0
    cy.caustics_reflective = cy.caustics_refractive = False
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    cy.device = "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            prefs.compute_device_type = kind
            prefs.refresh_devices()
            gpus = [d for d in prefs.devices if d.type == kind]
            if gpus:
                for d in prefs.devices:
                    d.use = d in gpus
                cy.device = "GPU"
                print(f"[render] Cycles device: {kind} {[d.name for d in gpus]}")
                return
    except Exception as exc:  # fall back to CPU
        print(f"[render] GPU setup failed ({exc}); using CPU")
    print("[render] Cycles device: CPU")


def set_stamp(sc, text):
    r = sc.render
    r.use_stamp = bool(text)
    if not text:
        return
    for attr in ("use_stamp_date", "use_stamp_time", "use_stamp_render_time", "use_stamp_frame", "use_stamp_frame_range",
                 "use_stamp_memory", "use_stamp_hostname", "use_stamp_camera", "use_stamp_lens", "use_stamp_scene",
                 "use_stamp_marker", "use_stamp_filename", "use_stamp_sequencer_strip"):
        setattr(r, attr, False)
    r.use_stamp_note = True
    r.stamp_note_text = text
    r.use_stamp_labels = False
    r.stamp_font_size = 16
    r.stamp_foreground = (1, 1, 1, 1)
    r.stamp_background = (0, 0, 0, 0.7)


def render_png(sc, path, camera, frame, res, stamp=None):
    sc.camera = bpy.data.objects[camera]
    sc.frame_set(frame)
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.filepath = str(path)
    set_stamp(sc, stamp)
    bpy.ops.render.render(write_still=True)
    return path


def _load_array(path):
    im = bpy.data.images.load(str(path))
    w, h = im.size
    arr = np.empty(w * h * 4, dtype=np.float32)
    im.pixels.foreach_get(arr)
    bpy.data.images.remove(im)
    return arr.reshape(h, w, 4)


def _save_jpg(arr, path):
    h, w = arr.shape[:2]
    im = bpy.data.images.new("out_" + Path(path).stem, w, h, alpha=False, float_buffer=False)
    im.pixels.foreach_set(np.ascontiguousarray(arr, dtype=np.float32).ravel())
    im.filepath_raw = str(path)
    im.file_format = "JPEG"
    im.save()
    bpy.data.images.remove(im)


def png_to_jpg(png, jpg, gray_from_red=False):
    arr = _load_array(png)
    if gray_from_red:   # the NIR rings are rendered as red light: the red channel is the reflectance preview
        arr[..., :3] = arr[..., :1]
    _save_jpg(arr, jpg)


def contact_sheet(pngs, jpg):
    tiles = [_load_array(p) for p in pngs]
    h, w = tiles[0].shape[:2]
    sheet = np.ones((2 * h, 2 * w, 4), dtype=np.float32)
    # Blender image rows run bottom-up: tiles 1,2 go to the top row
    for k, t in enumerate(tiles):
        r, c = divmod(k, 2)
        y0 = h * (1 - r)
        sheet[y0:y0 + h, c * w:(c + 1) * w] = t
    _save_jpg(sheet, jpg)


FRONT_BAR_NAMES = ("LIGHT_BAR_FRONT", "NIR_RING_F", "NIR_LEDS_F", "WHITE_PANEL_F_1", "WHITE_PANEL_F_2", "WHITE_LEDS_F_1", "WHITE_LEDS_F_2")
DOOR_NAMES = ("DOOR", "LINER_DOOR", "DOOR_HANDLE", "HINGE_BARREL_1", "HINGE_BARREL_2", "HINGE_LEAF_DOOR_1", "HINGE_LEAF_DOOR_2")
NIR_NOTE = "NIR framing / illumination preview (grayscale of a render lit by the 850 nm rings) - Blender does not model NIR reflectance"


def render_stills(out_dir):
    sc = bpy.context.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="capture_rig_render_"))
    ev = capture_timeline()
    configure_cycles(sc, 128)
    # 01 overview: door open, studio light + soft environment
    set_world_strength(0.25)
    with hidden(show=("STUDIO_KEY", "STUDIO_FILL")):
        p = render_png(sc, tmp / "01.png", "CAM_OVERVIEW", 1, (1600, 900))
    png_to_jpg(p, out_dir / "01_overview.jpg")
    set_world_strength(0.0)
    # 02 cutaway: door and the front light bar hidden, white LEDs on
    with hidden(names=DOOR_NAMES + FRONT_BAR_NAMES, show=("STUDIO_FILL",)):
        set_world_strength(0.15)
        p = render_png(sc, tmp / "02.png", "CAM_CUTAWAY", ev["still_rgb"][0], (1600, 900),
                       stamp="Cutaway: door and front light bar hidden (white LEDs on)")
    png_to_jpg(p, out_dir / "02_cutaway_front.jpg")
    set_world_strength(0.0)
    # 03 RGB camera view, white LEDs only
    sc.view_settings.exposure = EXPOSURE_RGB
    p = render_png(sc, tmp / "03.png", "CAM_RGB", ev["still_rgb"][0], (RES_X, RES_Y), stamp="CAM_RGB - white LEDs only")
    png_to_jpg(p, out_dir / "03_view_rgb.jpg")
    # 04 NIR camera view, 850 nm rings only, converted to grayscale
    sc.view_settings.exposure = EXPOSURE_NIR
    p = render_png(sc, tmp / "04.png", "CAM_NIR", ev["still_nir"][0], (RES_X, RES_Y), stamp=NIR_NOTE)
    png_to_jpg(p, out_dir / "04_view_nir.jpg", gray_from_red=True)
    # 05 contact sheet: faces 1-4 from CAM_RGB
    sc.view_settings.exposure = EXPOSURE_RGB
    tiles = [render_png(sc, tmp / f"05_{k}.png", "CAM_RGB", f, (960, 540), stamp=f"face {k + 1}") for k, f in enumerate(ev["still_rgb"])]
    contact_sheet(tiles, out_dir / "05_faces_rgb.jpg")
    # 06 calibration jig in place of the fruit
    with hidden(names=("MANGO",), show=("JIG_BODY", "JIG_PLATE")):
        p = render_png(sc, tmp / "06.png", "CAM_RGB", ev["still_rgb"][0], (RES_X, RES_Y), stamp="CAM_RGB - calibration jig (ArUco board at Z=150 mm)")
    png_to_jpg(p, out_dir / "06_calibration_jig.jpg")
    set_stamp(sc, None)
    sc.view_settings.exposure = 0.0
    sc.frame_set(1)
    for f in sorted(out_dir.glob("0*.jpg")):
        print(f"[render] {f.name} {f.stat().st_size // 1024} KiB")


def render_animation(out_dir):
    sc = bpy.context.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    ee = sc.eevee
    ee.taa_render_samples = 20
    ee.use_raytracing = True
    set_world_strength(ANIM_AMBIENT)
    sc.camera = bpy.data.objects["CAM_ANIM"]
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 960, 540, 100
    sc.frame_start, sc.frame_end = 1, FRAME_END
    set_stamp(sc, None)
    r = sc.render
    r.image_settings.file_format = "FFMPEG"
    r.ffmpeg.format = "MPEG4"
    r.ffmpeg.codec = "H264"
    r.ffmpeg.constant_rate_factor = "HIGH"
    r.ffmpeg.ffmpeg_preset = "GOOD"
    r.ffmpeg.audio_codec = "NONE"
    r.image_settings.color_mode = "RGB"
    r.filepath = str(out_dir / "capture_cycle")
    for old in out_dir.glob("capture_cycle*.mp4"):
        old.unlink()
    with hidden(names=("WALL_BACK", "LINER_BACK"), show=("BOUNCE_FILL",)):   # sectioned view: the back wall is removed
        bpy.ops.render.render(animation=True)
    set_world_strength(0.0)
    made = sorted(out_dir.glob("capture_cycle*.mp4"))
    if made and made[0].name != "capture_cycle.mp4":
        made[0].rename(out_dir / "capture_cycle.mp4")
    sc.frame_set(1)
    print(f"[anim] {out_dir / 'capture_cycle.mp4'}")


def export_all(out_dir, render=True, only=None):
    steps = set(only.split(",")) if only else {"blend", "stl", "csv", "stills", "anim"}
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    door = bpy.data.objects["DOOR"]
    if "csv" in steps:
        export_cut_list(out_dir / "mdf_cut_list.csv")
    if "stl" in steps:
        saved = door.rotation_euler.z
        door.rotation_euler.z = 0.0
        bpy.context.view_layer.update()
        export_stls(out_dir / "stl")
        door.rotation_euler.z = saved
        bpy.context.view_layer.update()
    if "blend" in steps:
        save_blend(out_dir / "capture_rig.blend")
    if render and "stills" in steps:
        render_stills(out_dir / "renders")
    if render and "anim" in steps:
        render_animation(out_dir / "renders")


# <<EXPORT_RENDER>>

# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None):
    argv = list(sys.argv if argv is None else argv)
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(prog="build_capture_rig.py", description=__doc__.split("\n")[0])
    p.add_argument("--check-only", action="store_true", help="build, run the design checks, exit 0/1")
    p.add_argument("--export-dir", help="write .blend, STLs, cut list and renders here")
    p.add_argument("--no-render", action="store_true", help="skip stills and the animation")
    p.add_argument("--skip-build", action="store_true", help="debug: run the checks on an empty scene")
    p.add_argument("--only", help="comma separated export steps: blend,stl,csv,stills,anim")
    return p.parse_args(argv)


def frame_overview_in_gui():
    """When run inside the Blender GUI: look at the rig from the front-left, high."""
    if bpy.app.background:
        return
    sc = bpy.context.scene
    sc.camera = bpy.data.objects.get("CAM_OVERVIEW", sc.camera)
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            space.clip_start, space.clip_end = 0.002, 20.0
            space.shading.type = "SOLID"
            space.shading.color_type = "MATERIAL"
            r3d = space.region_3d
            r3d.view_perspective = "PERSP"
            r3d.view_location = V(-20, -10, 170)
            r3d.view_distance = 1.25
            r3d.view_rotation = Euler((math.radians(62), 0.0, math.radians(-35)), "XYZ").to_quaternion()


def main():
    args = parse_args()
    if args.export_dir:
        ensure_aruco_board()
    if args.skip_build:
        clear_scene()
    else:
        build_scene()
    if args.check_only:
        failures = run_checks()
        if failures:
            for f in failures:
                print("FAIL", f)
            sys.stdout.flush()
            sys.exit(1)
        print("ALL CHECKS PASSED")
        sys.stdout.flush()
        sys.exit(0)
    if args.export_dir:
        failures = run_checks()
        for f in failures:
            print("FAIL", f)
        export_all(Path(args.export_dir), render=not args.no_render, only=args.only)
        if failures:
            sys.exit(1)
        return
    frame_overview_in_gui()


if "bpy" in sys.modules:
    main()
