#!/usr/bin/env python
import open3d as o3d
import open3d.visualization.gui as gui

class change_something:
    def __init__(self,pts):
        self.sigma = 0.0001
        self.pts = pts
    
    def on_key(self,e):
        if e.key == gui.KeyName.SPACE:
            if e.type == gui.KeyEvent.UP:  # check UP so we default to DOWN
                self.sigma += 0.0001
                print("[debug] SPACE released")
                print(self.sigma)
        return gui.Widget.EventCallbackResult.IGNORED    

def on_mouse(e):
    if e.type == gui.MouseEvent.Type.BUTTON_DOWN:
        print("[debug] mouse:", (e.x, e.y))
    return gui.Widget.EventCallbackResult.IGNORED

def on_key(e):
    if e.key == gui.KeyName.SPACE:
        if e.type == gui.KeyEvent.UP:  # check UP so we default to DOWN
            print("[debug] SPACE released")
        else:
            print("[debug] SPACE pressed")
        return gui.Widget.EventCallbackResult.HANDLED
    if e.key == gui.KeyName.W:  # eats W, which is forward in fly mode
        print("[debug] Eating W")
        return gui.Widget.EventCallbackResult.CONSUMED
    return gui.Widget.EventCallbackResult.IGNORED

def main():
    gui.Application.instance.initialize()
    w = gui.Application.instance.create_window("Open3D Example - Events",
                                               640, 480)
    scene = gui.SceneWidget()
    scene.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
    w.add_child(scene)

    # obj = o3d.geometry.TriangleMesh.create_morbius()
    # obj.compute_vertex_normals()
    pcd = o3d.io.read_point_cloud(f'/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper.ply')
    #pcd.compute_vertex_normals()

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    scene.scene.add_geometry("Point Cloud", pcd, material)

    scene.setup_camera(60, scene.scene.bounding_box, (0, 0, 0))
    # scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    cs = change_something()
    scene.set_on_mouse(on_mouse)
    scene.set_on_key(cs.on_key)

    gui.Application.instance.run()

if __name__ == "__main__":
    main()