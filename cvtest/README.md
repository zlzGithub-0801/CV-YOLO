# cvtest 文件说明
从`coco2017 validation`数据集随机挑选 60 张图片，并整合对应的 instance、caption 与 panoptic 标注信息。每个样本以一条 JSON 记录表示，整体存储为一个 JSON 数组。其中每一条的 key 包括：

### 1. image_id (number)
- COCO 数据集中该图像的唯一标识符

### 2. file_name (string)
- 图像文件名，对应 images/ 目录下的文件

### 3. width (number)
- 宽度（像素）

### 4. height (number)
- 高度（像素）

### 5. instances (dict array)
- 实例级目标检测与分割标注，为一个数组。数组中每一项为该图中检测到的一个实例。对应信息有：

#### 5.1 segmentation (array<number[]>)
- 实例的多边形分割轮廓，以 [x1, y1, x2, y2, ...] 形式表示。若为多段多边形，则为多个数组的列表。

#### 5.2 area (number)
- 该实例在图像中的像素面积。

#### 5.3 iscrowd (number)
- 是否为群体标注，0 表示单个实例，1 表示 crowd（如人群、密集物体）。

#### 5.4 image_id (number)
- 对应图像的 ID（应与顶层 image_id 相同）

#### 5.5 bbox (array<number>)
- 边界框，格式为：[x, y, width, height]，(x, y) 为左上角坐标

#### 5.6 category_id (number)
- 实例所属类别的 ID（COCO category），ID 和 category 具体对应信息可以在 `./categories.json` 中找到

#### 5.7 id (number)
- 该实例标注的唯一 ID

### 6. captions (dict array)
- 该图像对应的多条自然语言描述，为一个数组。数组中每一项为一个自然语言描述。

#### 6.1 image_id (number)
- 对应图像的 ID（应与顶层 image_id 相同）

#### 6.2 id (number)
- 该自然语言描述的唯一 ID

#### 6.3 caption (string)
- 自然语言描述内容。

### 7. panoptic (dict)
- 该图像的 panoptic segmentation 标注。包含以下信息：

#### 7.1 segments_info (array<object>)
- 图像中所有 segment 的信息列表，每个 segment 对应一个语义或实例区域

##### 7.1.1 id (number)
- segment 的唯一 ID（对应 panoptic PNG 中的像素值编码）

##### 7.1.2 category_id (number)
- segment 的类别 ID

##### 7.1.3 iscrowd (number)
- 是否为 crowd 区域

##### 7.1.4 bbox (array<number>)
- segment 的边界框

##### 7.1.5 area (number)
- segment 的像素面积

#### 7.2 file_name
- 对应 panoptic 分割标注文件名（PNG）（对应文件尚未上传，如有需要可以上传）

#### 7.3 image_id
- 对应图像的 ID（与顶层 image_id 相同）

主要关注instances和captions即可。后期可能添加stuff。