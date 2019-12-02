# FDMSE201904-MachineLearningHomeWork
FDMSE201904-机器学习课程作业

# 计划
- 11.30 app原型/视频剧本/论文大纲/ppt大纲

# 分工
- Amber: 故事策划/PPT/课堂汇报
- Brady: Code/App原型/相关的流程和说明文档
- Tjl:   视频录制和后期处理
- Panda: Word论文

# 进度

## amber
- 故事原型已经release

## brady
- [x] win10训练环境已调通
- [x] win10迁移学习已调通
- [x] app开发环境已调通
- [ ] ubuntu下模型转为手机可用的lite模型 进行中
- [x] 以tensorflow/example中的posnet例程为基础，根据故事原型调整appUI
- [x] app1.0版本已上传
- [x] app1.1版本已上传（加入了姿态判断算法）

# 版本说明

## 1.0版本
  - 硬编码，直接使用keypoint的相对位置进行判断
  
## 1.1版本 
  - 通过采样测试图片，进行数据人工分析之后，不再使用位置，而是使用角度进行判断
  - 设计了用于判断的策略矩阵，大大简化了代码量，效果优于1.0版本
  - 数据表的每一项为理想角度，上下容差范围设计为40度，具体数据见（celue-data.xlsx文件）




