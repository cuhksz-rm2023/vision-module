# vision-module
V0.1

图像处理方法
  {
      图像处理先将图片从RGB转成HSV；
      split，将s，v合并以过滤颜色（橙色红色较好辨别）；
      高斯模糊；
      canny边缘检测，作为目标的图片threshold参数暴力点，作为输入的图片threshold参数较温柔；
  }
匹配+数据过滤方法
  {
    使用ORB匹配；
    使用percentile（去除开头0%的数据，结尾10%的数据）过滤方法   #需提升
  }
  
  提升方向
    {
      导入视频之后变成啥样orz
      最好能排除远处圆圈的干扰，更精确的定位近处的圈；
      是否有方法，可以定位远处的圈作为二级目标，路径规划更高效；
      后期加上卡尔曼滤波预测，导航效果也许会更好？
    }
    
    
V0.2

{更新内容：使用红色和黄色mask对原图进行bitwise_or操作，再将其与【HSV图片提取的S,V合并样图】进行bitwise_and操作。
此项修改可以增强对远端圆环的锁定能力。}

{缺点：对终点红色环识别能力仍然不足。穿过圆环的时候会锁定到环边上。未锁定时会卡顿。锁定的时候圆心会不规则浮动。}

{改进方向：在识别不到圆环的时候，输出FALSE来让控制端采取【朝向下一个圆环飞行】的模式。并且抽取其中几帧作为导航就足够了，不然浮动太频繁不利于导航。提升输入图片的对比度和亮度，试图解决识别红圈的问题。进行图像坐标到相机坐标的转换，外参标定。}

V0.3


{更新内容：1、调整红色，黄色掩膜参数，使得红圈识别可行：通过对红色掩膜的输入图像进行增强对比度，增加亮度，调整红色掩膜色调，明亮度来过滤杂色。2、一旦无法找到中心，将会输出not found}

{缺点：不能远程锁定红圈，可以中，近距离锁定红圈。}

{待进行事项：外参标定}（本人线代还没学orz）

外参标定代码已由钟业斌学长完成（记录时间2023/5/19，在这之前就已完成，现在只是标注一下）


V0.4


{更新内容：1、使用霍夫圆变换识别目标环，替代原本的相似点识别算法，提升识别稳定性，降低穿环时锁环上的可能}

{缺点：偶尔还是会出现锁在环上的可能（一两帧）}

{待进行事项：添加数据过滤，避免输出错误坐标}


V0.4.1


{更新内容：添加了以方差为基础的数据过滤}
