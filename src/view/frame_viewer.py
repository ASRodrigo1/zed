import cv2


class Displayer(object):
    """
    Implements a class to display individual frames
    """

    def __init__(self, width: int, height: int, title: str = "Displayer") -> None:
        self.width = width
        self.height = height
        self.title = title

        # Create the window with the given name
        cv2.namedWindow(self.title)

    def draw_bbox(self, img, box, conf, label):
        """Draws bounding boxe in a frame"""
        ax, ay, cx, cy = int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])

        img = cv2.rectangle(img, (ax, ay), (cx, cy), (255, 0, 0), 2)
        img = cv2.putText(
            img,
            label + f": {conf:.3f}",
            (ax + 5, ay + 15),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

        return img

    def show(self, frame):
        """Show a frame in the window"""

        # Check if frame is in the correct shape
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        # Show the frame
        cv2.imshow(self.title, frame)

        # Wait for a command
        key = cv2.waitKey(1) & 0xFF

        return key
