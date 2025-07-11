# 🛠 SyncTable - 資料同步工具

**SQL Server ➜ PostgreSQL 批次資料同步解決方案**

本工具提供高效能的 SQL Server 到 PostgreSQL 資料同步能力，支援自動建表、型別轉換、定時排程及容器化部署。

## ✨ 功能特點

- ⛓ **自動連接** - 支援 SQL Server 與 PostgreSQL 雙向連接
- 🧱 **智慧建表** - 自動建立目標資料表並進行欄位型別轉換
- 🧼 **資料清理** - 自動處理空白字串與時間欄位格式化
- 🧹 **全量同步** - 自動清空 PostgreSQL 目標資料表（非增量備份）
- 🔁 **定時排程** - 支援可調整間隔的定時執行
- 🐳 **容器化** - 提供完整的 Docker 部署方案
- 📊 **批次處理** - 大量資料的效能優化控制
- 📝 **詳細日誌** - 完整的執行記錄與錯誤追蹤

## 🚀 快速開始

### 方式一：本機直接執行

1. **安裝依賴**
   ```bash
   pip install -r requirements.txt
   ```

2. **設定環境變數**
   ```bash
   cp .env.example .env
   # 編輯 .env 檔案
   ```

3. **執行程式**
   ```bash
   python data_sync.py
   ```

### 方式二：Docker Compose（推薦）

1. **一鍵啟動**
   ```bash
   docker-compose up -d
   ```

2. **查看日誌**
   ```bash
   docker-compose logs -f data-sync
   ```

3. **停止服務**
   ```bash
   docker-compose down
   ```

## ⚙️ 環境變數設定

創建 `.env` 檔案並設定以下參數：

```ini
# SQL Server 連線設定
SQL_SERVER_HOST=
SQL_SERVER_PORT=
SQL_SERVER_DATABASE=
SQL_SERVER_USERNAME=
SQL_SERVER_PASSWORD=

# PostgreSQL 連線設定
POSTGRESQL_HOST=
POSTGRESQL_PORT=
POSTGRESQL_DATABASE=
POSTGRESQL_USERNAME=
POSTGRESQL_PASSWORD=

# 同步資料表設定
SYNC_SOURCE_TABLE=
SYNC_TARGET_TABLE=
SYNC_CHUNKSIZE=1000
SYNC_BATCH_SIZE=10000

# 執行設定
EXECUTION_ENABLE_SCHEDULE=true
EXECUTION_SCHEDULE_INTERVAL=3600  # 單位：秒
```

### 📋 環境變數說明

| 變數名稱 | 說明 | 預設值 | 必填 |
|---------|------|----|------|
| `SQL_SERVER_HOST` | SQL Server 主機位址 | -  | ✅ |
| `SQL_SERVER_PORT` | SQL Server 連接埠 | -  | ✅ |
| `SQL_SERVER_DATABASE` | 來源資料庫名稱 | -  | ✅ |
| `SQL_SERVER_USERNAME` | SQL Server 使用者名稱 | -  | ✅ |
| `SQL_SERVER_PASSWORD` | SQL Server 密碼 | -  | ✅ |
| `POSTGRESQL_HOST` | PostgreSQL 主機位址 | -  | ✅ |
| `POSTGRESQL_PORT` | PostgreSQL 連接埠 | -  | ✅ |
| `POSTGRESQL_DATABASE` | 目標資料庫名稱 | -  | ✅ |
| `POSTGRESQL_USERNAME` | PostgreSQL 使用者名稱 | -  | ✅ |
| `POSTGRESQL_PASSWORD` | PostgreSQL 密碼 | -  | ✅ |
| `SYNC_SOURCE_TABLE` | 來源資料表 (schema.table) | -  | ✅ |
| `SYNC_TARGET_TABLE` | 目標資料表 (schema.table) | -  | ✅ |
| `SYNC_CHUNKSIZE` | 寫入批次大小 | - | ✅ |
| `SYNC_BATCH_SIZE` | 讀取批次大小 | - | ✅ |
| `EXECUTION_ENABLE_SCHEDULE` | 是否啟用定時執行 | - | ✅ |
| `EXECUTION_SCHEDULE_INTERVAL` | 執行間隔（秒） | - | ✅ |


## 📂 檔案結構

```
SyncTable/
├── data_sync.py              # 主程式
├── requirements.txt          # Python 依賴
├── Dockerfile               # Docker 映像檔
├── docker-compose.yml       # Docker Compose 配置
├── .env.example            # 環境變數範例
├── .env                    # 環境變數設定（需自行創建）
├── logs/                   # 日誌目錄
│   └── data_sync.log
└── README.md              # 說明文件
```

## 📊 日誌與監控

### 日誌位置
- **本機執行**：`./data_sync.log`
- **Docker 執行**：`./logs/data_sync.log`（掛載至主機）

### 日誌內容
- 連線狀態
- 同步進度
- 資料筆數統計
- 錯誤訊息
- 執行時間

### 監控指令
```bash
# 查看即時日誌
docker-compose logs -f data-sync

# 查看服務狀態
docker-compose ps

# 查看資源使用量
docker stats data_sync_service
```

## 🧪 驗證與測試

### SQL 驗證語法

**檢查來源資料**
```sql
-- SQL Server
SELECT COUNT(*) FROM dbo.source_table;
SELECT TOP 10 * FROM dbo.source_table;
```

### 連線測試
```bash
# 測試 SQL Server 連線
python -c "import pyodbc; print('SQL Server 驅動程式可用')"

# 測試 PostgreSQL 連線
python -c "import psycopg2; print('PostgreSQL 驅動程式可用')"
```

## ⚠️ 注意事項

### 資料同步特性
- 🔥 **全量同步**：每次執行都會 `TRUNCATE` 目標表
- 📝 **非增量**：不支援差異同步，適用於主資料表同步
- 🔒 **事務保護**：使用資料庫事務確保資料一致性

### 權限要求
**SQL Server**
- `SELECT` 權限於來源資料表
- `INFORMATION_SCHEMA` 查詢權限

**PostgreSQL**
- `CREATE TABLE` 權限
- `TRUNCATE` 權限
- `INSERT` 權限於目標 schema

### 效能建議
- 調整 `SYNC_BATCH_SIZE` 以平衡記憶體使用與效能
- 調整 `SYNC_CHUNKSIZE` 以優化寫入效能
- 根據資料量調整 `EXECUTION_SCHEDULE_INTERVAL`

## 🔧 故障排除

### 常見問題

**1. ODBC 驅動程式問題**
```bash
# 檢查可用驅動程式
python -c "import pyodbc; print(pyodbc.drivers())"
```

**2. 連線逾時**
- 檢查網路連通性
- 確認防火牆設定
- 驗證連線參數

**3. 權限不足**
- 確認資料庫使用者權限
- 檢查 schema 存取權限

**4. 記憶體不足**
- 降低 `SYNC_BATCH_SIZE`
- 增加 Docker 記憶體限制

## 📈 效能優化

### 建議設定

**小型資料表 (< 10萬筆)**
```ini
SYNC_CHUNKSIZE=1000
SYNC_BATCH_SIZE=5000
```

**中型資料表 (10萬 - 100萬筆)**
```ini
SYNC_CHUNKSIZE=2000
SYNC_BATCH_SIZE=10000
```

**大型資料表 (> 100萬筆)**
```ini
SYNC_CHUNKSIZE=5000
SYNC_BATCH_SIZE=20000
```

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

本專案採用 MIT 授權條款。

---

**作者**: AMO  
**聯絡**: rebith00723@gmail.com  
**版本**: 2025.07.03