# 手动坏盘故障演示 (Manual Disk Fault Demo)

## 环境限制与前置条件

*   **操作系统**: 必须是 **Linux** 系统。
    *   本演示依赖 Linux 内核模块 `scsi_debug` 和 Device Mapper (`dm-mod`)。
    *   **不支持** MacOS (Darwin) 或 Windows。
*   **权限**: 所有脚本执行都需要 `root` 权限 (通常使用 `sudo`)，因为涉及内核模块加载、设备创建和挂载操作。
*   **系统工具依赖**:
    *   `dmsetup`: 用于管理 Device Mapper 设备。
    *   `mkfs.ext4`: 用于格式化磁盘。
    *   `journalctl`: 用于采集内核日志。
    *   `modprobe`: 用于加载内核模块。

### 依赖安装指南

在不同的 Linux 发行版上，你可以使用包管理器安装上述依赖：

**Debian / Ubuntu / Kali Linux**:
```bash
sudo apt-get update
sudo apt-get install -y dmsetup e2fsprogs util-linux systemd kmod
```

**CentOS / RHEL / Fedora**:
```bash
sudo yum install -y device-mapper e2fsprogs util-linux systemd kmod
# 或使用 dnf
sudo dnf install -y device-mapper e2fsprogs util-linux systemd kmod
```

**Alpine Linux**:
```bash
sudo apk add device-mapper e2fsprogs util-linux systemd kmod
```

**验证安装**:
确保以下命令能正常执行（可能需要 sudo）：
```bash
sudo dmsetup version
mkfs.ext4 -V
modprobe --version
```

这个目录包含了一组脚本，用于演示如何在 Linux 系统上模拟“坏盘”故障，并观察业务应用的反应。
这一套工具将环境准备、业务运行、故障注入和恢复分离开来，方便用户手动控制流程。

## 目录结构

*   `common.py`: 公共辅助函数。
*   `setup_disk.py`: 准备虚拟磁盘环境（创建设备、格式化、挂载）。
*   `business_app.py`: 模拟业务应用，持续对磁盘进行读写操作。
*   `inject_fault.py`: 注入坏盘故障（将磁盘切换为错误状态）。
*   `recover_fault.py`: 恢复磁盘（将磁盘切换回正常状态）。
*   `teardown.py`: 清理环境（卸载磁盘、移除设备）。

## 使用步骤

### 1. 准备环境

首先，运行 `examples/manual_disk_fault/setup_disk.py` 来创建虚拟磁盘并挂载。
注意：需要 `sudo` 权限。

```bash
python examples/manual_disk_fault/setup_disk.py
```

成功后，你会看到类似输出：
```
SUCCESS: Disk setup complete.
Device: /dev/mapper/demo_disk
Mounted at: /mnt/demo_disk
```

### 2. 启动业务应用

在一个单独的终端窗口中，运行业务应用。它会持续写入和读取数据。

```bash
python examples/manual_disk_fault/business_app.py
```

你会看到正常的日志输出：
```
[INFO] Transaction 1 committed successfully.
[INFO] Transaction 2 committed successfully.
...
```

### 3. 注入故障

保持业务应用运行，在另一个终端窗口中运行故障注入脚本：

```bash
python examples/manual_disk_fault/inject_fault.py
```

脚本执行完毕后，回到业务应用的终端，你会立即看到报错信息：
```
[ERROR] Transaction 15 FAILED! Disk Error: [Errno 5] Input/output error
```

同时，当前目录下会生成 `kernel.log` 文件，记录了内核层面的错误日志。

此时磁盘的所有 I/O 操作都会失败。

### 4. 恢复故障 (可选)

如果你想让磁盘恢复正常，可以运行：

```bash
python examples/manual_disk_fault/recover_fault.py
```

恢复后，业务应用的日志将变回正常状态。

### 5. 清理环境

测试结束后，停止业务应用 (Ctrl+C)，然后运行清理脚本：

```bash
python examples/manual_disk_fault/teardown.py
```

这会卸载文件系统并移除虚拟设备。
