import rclpy
from rclpy.node import Node

class NuScenesPlayer(Node):
    def __init__(self):
        super().__init__('nuscenes_player')
        self.get_logger().info('NuScenes Player Node has been started.')



def main(args=None):
    rclpy.init(args=args)
    node = NuScenesPlayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
