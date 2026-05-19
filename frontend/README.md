# 组合视频检索系统前端

这是一个无依赖静态前端，直接对接接口说明中的两个后端接口：

- `GET /api/videos?cursor=&limit=24`
- `POST /api/search`

## 运行

在 `frontend` 目录启动任意静态服务器即可，例如：

```powershell
python -m http.server 5173
```

打开：

```text
http://localhost:5173/
```

如果后端与前端不在同一域名，可以在 `index.html` 加载脚本前注入：

```html
<script>
  window.APP_CONFIG = {
    API_BASE: "http://localhost:8000"
  };
</script>
```

本地无后端时可用演示数据预览界面：

```text
http://localhost:5173/?mock=1
```

## 已实现

- 视频库游标分页与参考视频选择
- 修改文本、保留内容、排除内容、返回数量、调试开关
- `/api/search` 请求、加载态、错误态、保留旧结果
- 低置信结果降权和全部低置信提示
- 调试分支分数条、动作类别、原始分数
- 结果缩略图弹窗预览
- 桌面与移动端响应式布局
