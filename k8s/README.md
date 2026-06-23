# Kubernetes 部署配置 - 专业课程答疑系统

本目录包含课程答疑系统的 Kubernetes 部署配置文件。

## 文件说明

- `namespace.yaml` - 命名空间配置
- `configmap.yaml` - 应用配置
- `secret.yaml` - 敏感信息（API 密钥等）
- `deployment.yaml` - 应用部署配置
- `service.yaml` - 服务暴露配置
- `ingress.yaml` - 入口配置（可选）
- `hpa.yaml` - 水平自动扩缩容配置

## 快速部署

```bash
# 1. 创建命名空间
kubectl apply -f namespace.yaml

# 2. 创建配置和密钥
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml

# 3. 部署应用
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# 4. （可选）配置入口
kubectl apply -f ingress.yaml

# 5. （可选）配置自动扩缩容
kubectl apply -f hpa.yaml
```

## 配置说明

### 敏感信息

在部署前，需要修改 `secret.yaml` 中的 API 密钥：

```yaml
data:
  DEEPSEEK_API_KEY: <base64-encoded-key>
  BAILIAN_API_KEY: <base64-encoded-key>
```

使用以下命令生成 base64 编码：

```bash
echo -n "your-api-key" | base64
```

### 配置调整

根据实际需求调整 `configmap.yaml` 中的配置参数。

## 验证部署

```bash
# 查看 Pod 状态
kubectl get pods -n course-qa

# 查看服务
kubectl get svc -n course-qa

# 查看日志
kubectl logs -n course-qa -l app=course-qa

# 测试健康检查
kubectl port-forward -n course-qa svc/course-qa 5000:5000
curl http://localhost:5000/health
```

## 扩缩容

```bash
# 手动扩容
kubectl scale deployment course-qa -n course-qa --replicas=3

# 查看 HPA 状态
kubectl get hpa -n course-qa
```

## 更新部署

```bash
# 更新镜像
kubectl set image deployment/course-qa course-qa=your-registry/course-qa:v2 -n course-qa

# 滚动更新状态
kubectl rollout status deployment/course-qa -n course-qa

# 回滚
kubectl rollout undo deployment/course-qa -n course-qa
```
