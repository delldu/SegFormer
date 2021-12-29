"""Demo."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 06月 17日 星期四 14:09:56 CST
# ***
# ************************************************************************************/
#

import segment_former

if __name__ == "__main__":
    # segment_former.image_predict("images/*.png", "output")
    # segment_former.image_client("PAI", "images/*.png", "output")
    # segment_former.image_server("PAI")

    segment_former.video_client("PAI", "/home/dell/tennis.mp4", "output/tennis.mp4")
    segment_former.video_server("PAI")
