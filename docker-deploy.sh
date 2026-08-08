#!/bin/bash

# EchoMind 智能客服系统 - Docker 部署脚本


set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
PROJECT_NAME="echomind"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# 函数：打印信息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 函数：检查依赖
check_dependencies() {
    print_info "检查依赖..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi

    print_info "依赖检查完成"
}

# 函数：创建必要的目录
create_directories() {
    print_info "创建必要的目录..."

    mkdir -p data/chroma
    mkdir -p logs
    mkdir -p config/nginx/ssl
    mkdir -p config/grafana/provisioning
    mkdir -p config/grafana/dashboards
    mkdir -p config/alerts

    print_info "目录创建完成"
}

# 校验指定环境文件中的关键生产凭证，不执行或 source 文件内容。
validate_env_file() {
    local env_path=$1
    local anthropic_key redis_password secret_key jwt_secret_key
    anthropic_key=$(awk -F= '/^ANTHROPIC_API_KEY=/{sub(/^[^=]*=/, ""); print; exit}' "$env_path")
    redis_password=$(awk -F= '/^REDIS_PASSWORD=/{sub(/^[^=]*=/, ""); print; exit}' "$env_path")
    secret_key=$(awk -F= '/^SECRET_KEY=/{sub(/^[^=]*=/, ""); print; exit}' "$env_path")
    jwt_secret_key=$(awk -F= '/^JWT_SECRET_KEY=/{sub(/^[^=]*=/, ""); print; exit}' "$env_path")
    if [ -z "$anthropic_key" ] || [[ "$anthropic_key" == your-* ]]; then
        print_error "ANTHROPIC_API_KEY 未配置"
        return 1
    fi
    if [ -z "$redis_password" ] || [[ "$redis_password" == replace-* ]] || [ ${#redis_password} -lt 16 ]; then
        print_error "REDIS_PASSWORD 必须设置为至少 16 位的随机密码"
        return 1
    fi
    if [ -z "$secret_key" ] || [[ "$secret_key" == change_this* ]] || [ ${#secret_key} -lt 32 ]; then
        print_error "SECRET_KEY 必须设置为至少 32 位的随机密钥"
        return 1
    fi
    if [ -z "$jwt_secret_key" ] || [[ "$jwt_secret_key" == change_this* ]] || [ ${#jwt_secret_key} -lt 32 ]; then
        print_error "JWT_SECRET_KEY 必须设置为至少 32 位的随机密钥"
        return 1
    fi
}

# 函数：检查环境变量
check_env_file() {
    print_info "检查环境变量配置..."

    if [ ! -f "$ENV_FILE" ]; then
        print_warn ".env 文件不存在，从 .env.example 创建..."

        local example_file=""
        if [ -f ".env.example" ]; then
            example_file=".env.example"
        elif [ -f ".env.example.env" ]; then
            example_file=".env.example.env"
        fi

        if [ -n "$example_file" ]; then
            cp "$example_file" .env
            print_info "已创建 .env 文件，请编辑配置"
            print_error "请设置 ANTHROPIC_API_KEY 和强随机 REDIS_PASSWORD 后重新运行"
            exit 1
        else
            print_error "环境变量示例文件不存在"
            exit 1
        fi
    else
        print_info "环境变量配置文件已存在"
    fi

    if ! validate_env_file "$ENV_FILE"; then
        exit 1
    fi
}

# 函数：构建镜像
build_images() {
    print_info "构建 Docker 镜像..."

    docker-compose build --no-cache

    print_info "镜像构建完成"
}

# 函数：启动服务
start_services() {
    print_info "启动服务..."

    docker-compose up -d

    print_info "服务启动完成"
}

# 函数：停止服务
stop_services() {
    print_info "停止服务..."

    docker-compose down

    print_info "服务已停止"
}

# 函数：重启服务
restart_services() {
    print_info "重启服务..."

    docker-compose restart

    print_info "服务已重启"
}

# 函数：查看服务状态
status_services() {
    print_info "服务状态:"

    docker-compose ps
}

# 函数：查看日志
view_logs() {
    local service=$1

    if [ -z "$service" ]; then
        print_info "查看所有服务日志..."
        docker-compose logs -f
    else
        print_info "查看 $service 服务日志..."
        docker-compose logs -f "$service"
    fi
}

# 函数：健康检查
health_check() {
    print_info "执行健康检查..."

    # 等待服务启动
    sleep 10

    # 检查主应用
    if curl -sf http://localhost:8000/health > /dev/null; then
        print_info "✓ 主应用健康"
    else
        print_error "✗ 主应用不健康"
    fi

    # 检查 Redis
    if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
        print_info "✓ Redis 健康"
    else
        print_error "✗ Redis 不健康"
    fi

    # 检查 ChromaDB
    if curl -sf http://localhost:8001/api/v1/heartbeat > /dev/null; then
        print_info "✓ ChromaDB 健康"
    else
        print_error "✗ ChromaDB 不健康"
    fi

    # 检查 Prometheus
    if curl -sf http://localhost:9090/-/healthy > /dev/null; then
        print_info "✓ Prometheus 健康"
    else
        print_error "✗ Prometheus 不健康"
    fi
}

# 函数：清理资源
cleanup() {
    print_warn "清理所有资源（包括数据卷）..."

    read -p "确认清理？这将删除所有数据 (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v
        print_info "清理完成"
    else
        print_info "清理已取消"
    fi
}

# 函数：备份数据
backup_data() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"

    print_info "备份数据到 $backup_dir..."

    mkdir -p "$backup_dir"

    # 备份 Redis 数据
    docker-compose exec -T redis redis-cli SAVE
    docker cp echomind-redis:/data/dump.rdb "$backup_dir/"

    # 备份 ChromaDB 数据
    docker cp echomind-chromadb:/chroma/chroma "$backup_dir/"

    # 备份配置
    cp .env "$backup_dir/"
    cp -r config "$backup_dir/"

    print_info "备份完成: $backup_dir"
}

# 函数：恢复数据
restore_data() {
    local backup_dir=$1

    if [ -z "$backup_dir" ]; then
        print_error "请指定备份目录"
        exit 1
    fi

    if [ ! -d "$backup_dir" ]; then
        print_error "备份目录不存在: $backup_dir"
        exit 1
    fi

    for required in dump.rdb .env chroma config; do
        if [ ! -e "$backup_dir/$required" ]; then
            print_error "备份不完整，缺少: $backup_dir/$required"
            exit 1
        fi
    done
    if [ ! -f "$backup_dir/dump.rdb" ] || [ ! -f "$backup_dir/.env" ] || \
       [ ! -d "$backup_dir/chroma" ] || [ ! -d "$backup_dir/config" ]; then
        print_error "备份中类型不正确 (dump.rdb/.env 应为文件, chroma/config 应为目录)"
        exit 1
    fi

    # 在停止服务前完整复制配置，先证明备份可读且目标有足够空间。
    local stage_dir recovery_dir
    stage_dir=$(mktemp -d "./restore-stage.XXXXXX")
    recovery_dir="backups/pre_restore_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$stage_dir/config"
    if ! cp "$backup_dir/.env" "$stage_dir/.env" || \
       ! cp -R "$backup_dir/config/." "$stage_dir/config/"; then
        print_error "备份配置预检失败，未修改当前数据"
        rm -rf "$stage_dir"
        exit 1
    fi
    if ! validate_env_file "$stage_dir/.env"; then
        print_error "备份环境变量未通过安全校验，未修改当前服务"
        rm -rf "$stage_dir"
        exit 1
    fi

    print_warn "从 $backup_dir 恢复数据..."
    read -p "确认恢复？这将覆盖现有数据 (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 停止服务
        docker-compose stop

        # 恢复服务数据；失败时保留当前配置并立即重新启动原服务。
        if ! docker cp "$backup_dir/dump.rdb" echomind-redis:/data/ || \
           ! docker cp "$backup_dir/chroma" echomind-chromadb:/chroma/; then
            print_error "服务数据恢复失败，当前配置未修改"
            rm -rf "$stage_dir"
            docker-compose start || true
            exit 1
        fi

        # 原配置先移动到可恢复目录，再原子切换已预检的新配置。
        mkdir -p "$recovery_dir"
        [ ! -e .env ] || mv .env "$recovery_dir/.env"
        [ ! -e config ] || mv config "$recovery_dir/config"
        if ! mv "$stage_dir/.env" .env || ! mv "$stage_dir/config" config; then
            print_error "配置切换失败，正在恢复原配置"
            [ ! -e .env ] || mv .env "$stage_dir/failed.env"
            [ ! -e config ] || mv config "$stage_dir/failed-config"
            [ ! -e "$recovery_dir/.env" ] || mv "$recovery_dir/.env" .env
            [ ! -e "$recovery_dir/config" ] || mv "$recovery_dir/config" config
            docker-compose start || true
            exit 1
        fi
        rmdir "$stage_dir"

        # 启动服务
        docker-compose start

        print_info "恢复完成；原配置保存在 $recovery_dir"
    else
        rm -rf "$stage_dir"
        print_info "恢复已取消"
    fi
}

# 函数：显示帮助信息
show_help() {
    cat << EOF
EchoMind 智能客服系统 - Docker 部署脚本

用法: ./docker-deploy.sh [命令]

命令:
    install     初始化安装（检查依赖、创建目录、构建镜像）
    start       启动所有服务
    stop        停止所有服务
    restart     重启所有服务
    status      查看服务状态
    logs        查看服务日志（可选指定服务名）
    health      执行健康检查
    build       重新构建镜像
    cleanup     清理所有资源（包括数据）
    backup      备份数据
    restore     恢复数据（需指定备份目录）
    help        显示此帮助信息

示例:
    ./docker-deploy.sh install
    ./docker-deploy.sh start
    ./docker-deploy.sh logs echomind
    ./docker-deploy.sh backup
    ./docker-deploy.sh restore backups/20231201_120000

环境变量:
    在 .env 文件中配置相关参数

EOF
}

# 主函数
main() {
    case "${1:-help}" in
        install)
            check_dependencies
            check_env_file
            create_directories
            build_images
            print_info "安装完成！运行 './docker-deploy.sh start' 启动服务"
            ;;
        start)
            check_env_file
            start_services
            health_check
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            status_services
            ;;
        logs)
            view_logs "$2"
            ;;
        health)
            health_check
            ;;
        build)
            build_images
            ;;
        cleanup)
            cleanup
            ;;
        backup)
            backup_data
            ;;
        restore)
            restore_data "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
