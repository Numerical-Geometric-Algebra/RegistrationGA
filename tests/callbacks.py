import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from open3d.visualization import gui
import open3d as o3d

class Open3DViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Open3D Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Create a simple point cloud
        points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
        pcd = o3d.geometry.PointCloud(points)

        # Create Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(pcd)

        # Embed Open3D rendering into the PyQt5 GUI
        self.container = QWidget(self)
        self.setCentralWidget(self.container)
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.vis.get_gui_widget(), 1)

        # Timer for continuous camera movement
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(16)  # Set the timer interval in milliseconds (60 FPS)

        # Connect keyboard events
        self.container.setFocusPolicy(Qt.StrongFocus)
        self.container.keyPressEvent = self.keyPressEvent

        # Run the visualizer
        self.vis.run()

    def keyPressEvent(self, event):
        step_size = 0.1  # Adjust the step size as needed
        camera = self.vis.get_view_control().convert_to_pinhole_camera_parameters()

        if event.key() == Qt.Key_W:
            camera.eye[2] -= step_size
        elif event.key() == Qt.Key_S:
            camera.eye[2] += step_size
        elif event.key() == Qt.Key_A:
            camera.eye[0] -= step_size
        elif event.key() == Qt.Key_D:
            camera.eye[0] += step_size

        self.vis.get_view_control().convert_from_pinhole_camera_parameters(camera)

    def update_camera(self):
        # Update the Open3D visualizer
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    def closeEvent(self, event):
        self.vis.destroy_window()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = Open3DViewer()
    window.show()
    app.exec_()