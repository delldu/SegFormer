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

import image_segment

if __name__ == "__main__":
    image_segment.image_predict("images/*.png", "output")
