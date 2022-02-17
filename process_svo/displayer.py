import cv2


class Displayer(object):
    """
    Implements a class to display individual frames and context
    """
    def __init__(self, width: int, height: int, title :str='Displayer') -> None:
        self.width = width
        self.height = height
        self.title = title

        # Create the window with the given name
        cv2.namedWindow(self.title)
    
    def show(self, frame) -> None:
        """Show a frame in the window"""
        
        # Check if frame is in the correct shape
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        
        # Show the frame
        cv2.imshow(self.title, frame)
        
        # Wait for a command
        key = cv2.waitKey(1) & 0xFF

        # Return the command
        return key