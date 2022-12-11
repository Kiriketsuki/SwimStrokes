import sys
import cv2
import os
from PyQt5 import QtWidgets, QtGui, QtCore, QtMultimedia

class DragDropWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        # Set the window title and dimensions
        self.setWindowTitle('Video Splitter')
        # Set to 800 by 600
        self.resize(800, 600)

        # Create a label for displaying dropped files
        self.label = QtWidgets.QLabel('Drop Video Files Here', self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Set label size to take up the entire window
        self.label.resize(self.width(), self.height())
        
        # Increase font size
        font = self.label.font()
        font.setPointSize(20)
        self.label.setFont(font)


        # Set the label as the target for drag and drop operations
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        # Check if the data being dragged is a file
        if event.mimeData().hasUrls():
            # Get the list of files being dragged
            urls = event.mimeData().urls()
            files = [u.toLocalFile() for u in urls]

            # Check if the files have a video file extension
            video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv']
            is_video_file = any(f.lower().endswith(ext) for ext in video_extensions for f in files)

            if is_video_file:
                # Accept the drag operation if the files are video files
                event.accept()
            else:
                # Reject the drag operation if the files are not video files
                event.ignore()
                # Display a warning in the label
                self.label.setText('Only Video Files Are Allowed')
        else:
            # Ignore the drag operation if the data is not a file
            event.ignore()

    def dropEvent(self, event):
        # Get the list of files being dropped
        urls = event.mimeData().urls()
        files = [u.toLocalFile() for u in urls]

        # Display the list of files in the label
        self.label.setText('\n'.join(files))

        # Display the first frame of the first video file
        # self.display_frame(files[0], self.label)

        # Save the frames of the first video file
        self.save_frames(files[0])

        self.show_popup()

    def display_frame(self, video_file, label):
    # Open the video file and read the first frame
        vid = cv2.VideoCapture(video_file)
        ret, frame = vid.read()

        # Convert the frame to a QImage object
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)

        # Create a QPixmap from the QImage and set it as the pixmap of the label
        pixmap = QtGui.QPixmap.fromImage(frame)
        label.setPixmap(pixmap)

    def save_frames(self, video_file):
        # Create a directory for the video frames
        dirname = os.path.splitext(os.path.basename(video_file))[0]
        os.makedirs(os.path.join('inputs', dirname), exist_ok=True)

        # Open the video file
        vid = cv2.VideoCapture(video_file)

        # Read the frames from the video and save them as images in the new directory
        i = 0
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                # Save the frame as an image
                cv2.imwrite(os.path.join('inputs', dirname, f'frame_{i:04d}.png'), frame)
                i += 1
            else:
                break

    def show_popup(self):
    # Create a message box with the "Information" icon and the "OK" button
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, 'Processing Complete', 'The video frames have been saved to the "inputs" directory.', QtWidgets.QMessageBox.Ok)

        # Show the message box and clear the file label when the "OK" button is clicked
        msg.exec_()
        self.label.clear()
        self.label.setText('Drop Video Files Here')




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = DragDropWindow()
    window.show()
    sys.exit(app.exec_())
