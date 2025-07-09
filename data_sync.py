import pandas as pd
import pyodbc
from sqlalchemy import create_engine, text
import logging
from datetime import datetime
import sys
import time
import signal
import os

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_sync.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============== 從環境變數讀取設定 ===============
def get_required_env_var(var_name):
    """從環境變數取得必要的設定值，如果不存在則拋出錯誤"""
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"必要的環境變數 '{var_name}' 未設定")
    return value


def get_optional_env_var(var_name, default_value, value_type=str):
    """從環境變數取得可選的設定值，如果不存在則使用預設值"""
    value = os.getenv(var_name)
    if value is None:
        return default_value

    # 根據類型轉換
    if value_type == int:
        try:
            return int(value)
        except ValueError:
            logger.warning(f"環境變數 '{var_name}' 的值 '{value}' 無法轉換為整數，使用預設值 {default_value}")
            return default_value
    elif value_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    else:
        return value


# SQL Server 連線設定 - 從環境變數讀取
SQL_SERVER_CONFIG = {
    'host': get_required_env_var('SQL_SERVER_HOST'),
    'port': int(get_required_env_var('SQL_SERVER_PORT')),
    'database': get_required_env_var('SQL_SERVER_DATABASE'),
    'username': get_required_env_var('SQL_SERVER_USERNAME'),
    'password': get_required_env_var('SQL_SERVER_PASSWORD')
}

# PostgreSQL 連線設定 - 從環境變數讀取
POSTGRESQL_CONFIG = {
    'host': get_required_env_var('POSTGRESQL_HOST'),
    'port': int(get_required_env_var('POSTGRESQL_PORT')),
    'database': get_required_env_var('POSTGRESQL_DATABASE'),
    'username': get_required_env_var('POSTGRESQL_USERNAME'),
    'password': get_required_env_var('POSTGRESQL_PASSWORD')
}

# 同步設定 - 從環境變數讀取
SYNC_CONFIG = {
    'source_table': get_required_env_var('SYNC_SOURCE_TABLE'),
    'target_table': get_required_env_var('SYNC_TARGET_TABLE'),
    'chunksize': int(get_required_env_var('SYNC_CHUNKSIZE')),
    'batch_size': int(get_required_env_var('SYNC_BATCH_SIZE'))
}

# 執行設定 - 從環境變數讀取（移除重試相關設定）
EXECUTION_CONFIG = {
    'schedule_interval': int(get_required_env_var('EXECUTION_SCHEDULE_INTERVAL')),
    'enable_schedule': get_required_env_var('EXECUTION_ENABLE_SCHEDULE').lower() in ('true', '1', 'yes', 'on')
}

# 在程式啟動時顯示設定資訊（隱藏密碼）

logger.info(r"""


╭──────────────────────────────────────────────╮
│                 SYNC_TABLE                   │
│                   By AMO                     │
│            rebith00723@gmail.com             │
│                 2025/07/03                   │
╰──────────────────────────────────────────────╯


""")
logger.info("=== 程式設定資訊 ===")
logger.info(f"SQL Server: {SQL_SERVER_CONFIG['host']}:{SQL_SERVER_CONFIG['port']}/{SQL_SERVER_CONFIG['database']}")
logger.info(f"PostgreSQL: {POSTGRESQL_CONFIG['host']}:{POSTGRESQL_CONFIG['port']}/{POSTGRESQL_CONFIG['database']}")
logger.info(f"同步設定: {SYNC_CONFIG['source_table']} -> {SYNC_CONFIG['target_table']}")
logger.info(f"批次設定: chunksize={SYNC_CONFIG['chunksize']}, batch_size={SYNC_CONFIG['batch_size']}")
logger.info(
    f"定期執行: {'啟用' if EXECUTION_CONFIG['enable_schedule'] else '停用'}, interval={EXECUTION_CONFIG['schedule_interval']}s")


class GracefulKiller:
    """優雅停止程式的處理器"""

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"接收到停止信號 {signum}，程式將在下次同步完成後停止...")
        self.kill_now = True


# =============== 連線字串建立 ===============
def get_available_odbc_driver():
    """取得可用的 ODBC 驅動程式"""
    drivers = pyodbc.drivers()
    logger.info(f"可用的 ODBC 驅動程式: {drivers}")

    # 優先順序：較新版本的驅動程式優先
    preferred_drivers = [
        "FreeTDS",
        "ODBC Driver 17 for SQL Server"
    ]

    for driver in preferred_drivers:
        if driver in drivers:
            logger.info(f"使用 ODBC 驅動程式: {driver}")
            return driver

    # 如果沒有找到偏好的驅動程式，使用第一個可用的
    if drivers:
        logger.warning(f"未找到偏好的驅動程式，使用: {drivers[0]}")
        return drivers[0]

    raise Exception("未找到任何可用的 ODBC 驅動程式")


def create_sql_server_connection():
    """建立 SQL Server 連線"""
    try:
        # 取得可用的 ODBC 驅動程式
        driver = get_available_odbc_driver()

        # 使用 pyodbc 連線字串
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={SQL_SERVER_CONFIG['host']},{SQL_SERVER_CONFIG['port']};"
            f"DATABASE={SQL_SERVER_CONFIG['database']};"
            f"UID={SQL_SERVER_CONFIG['username']};"
            f"PWD={SQL_SERVER_CONFIG['password']};"
            f"Encrypt=no;"
        )

        # 建立 SQLAlchemy engine
        from urllib.parse import quote_plus
        conn_str_encoded = quote_plus(conn_str)
        engine_str = f"mssql+pyodbc:///?odbc_connect={conn_str_encoded}"
        engine = create_engine(engine_str)

        # 測試連線
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.info("SQL Server 連線建立成功")
        return engine
    except Exception as e:
        logger.error(f"SQL Server 連線失敗: {e}")
        raise


def create_postgresql_connection():
    """建立 PostgreSQL 連線"""
    try:
        # 建立 SQLAlchemy engine
        engine_str = (
            f"postgresql://{POSTGRESQL_CONFIG['username']}:"
            f"{POSTGRESQL_CONFIG['password']}@"
            f"{POSTGRESQL_CONFIG['host']}:{POSTGRESQL_CONFIG['port']}/"
            f"{POSTGRESQL_CONFIG['database']}"
        )
        engine = create_engine(engine_str)

        # 測試連線
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.info("PostgreSQL 連線建立成功")
        return engine
    except Exception as e:
        logger.error(f"PostgreSQL 連線失敗: {e}")
        raise


# =============== 資料同步函數 ===============
def get_table_row_count(source_engine, source_table):
    """取得來源資料表的總筆數"""
    try:
        query = f"SELECT COUNT(*) as count FROM {source_table}"
        with source_engine.connect() as conn:
            result = conn.execute(text(query))
            count = result.fetchone()[0]
        logger.info(f"來源資料表 {source_table} 總筆數: {count:,}")
        return count
    except Exception as e:
        logger.error(f"取得資料表筆數失敗: {e}")
        raise


def truncate_target_table(target_engine, target_table):
    """清空目標資料表"""
    try:
        with target_engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {target_table}"))
        logger.info(f"目標資料表 {target_table} 已清空")
    except Exception as e:
        logger.error(f"清空目標資料表失敗: {e}")
        raise


def get_sql_server_table_schema(source_engine, source_table):
    """從 SQL Server 擷取資料表結構"""
    try:
        # 解析來源表的 schema 和 table 名稱
        if '.' in source_table:
            source_schema, source_table_name = source_table.split('.', 1)
        else:
            source_schema = 'dbo'
            source_table_name = source_table

        # 查詢資料表結構的 SQL
        schema_query = f"""
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            IS_NULLABLE,
            NUMERIC_PRECISION,
            NUMERIC_SCALE,
            DATETIME_PRECISION
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = '{source_schema}' 
        AND TABLE_NAME = '{source_table_name}'
        ORDER BY ORDINAL_POSITION
        """

        with source_engine.connect() as conn:
            result = conn.execute(text(schema_query))
            columns = result.fetchall()

        logger.info(f"從 SQL Server 擷取到 {len(columns)} 個欄位的結構資訊")
        return columns

    except Exception as e:
        logger.error(f"擷取 SQL Server 資料表結構失敗: {e}")
        raise


def convert_sql_server_to_postgresql_type(data_type, max_length, precision, scale):
    """將 SQL Server 資料類型轉換為 PostgreSQL 資料類型"""
    data_type = data_type.lower()

    type_mapping = {
        'varchar': f'VARCHAR({max_length})' if max_length and max_length != -1 else 'TEXT',
        'nvarchar': f'VARCHAR({max_length})' if max_length and max_length != -1 else 'TEXT',
        'char': f'CHAR({max_length})' if max_length else 'CHAR(1)',
        'nchar': f'CHAR({max_length})' if max_length else 'CHAR(1)',
        'text': 'TEXT',
        'ntext': 'TEXT',
        'int': 'INTEGER',
        'bigint': 'BIGINT',
        'smallint': 'SMALLINT',
        'tinyint': 'SMALLINT',
        'bit': 'BOOLEAN',
        'decimal': f'DECIMAL({precision},{scale})' if precision and scale else 'DECIMAL',
        'numeric': f'NUMERIC({precision},{scale})' if precision and scale else 'NUMERIC',
        'float': 'DOUBLE PRECISION',
        'real': 'REAL',
        'money': 'DECIMAL(19,4)',
        'smallmoney': 'DECIMAL(10,4)',
        'datetime': 'TIMESTAMP',
        'datetime2': 'TIMESTAMP',
        'smalldatetime': 'TIMESTAMP',
        'date': 'DATE',
        'time': 'TIME',
        'timestamp': 'BYTEA',
        'uniqueidentifier': 'UUID',
        'xml': 'XML',
        'varbinary': 'BYTEA',
        'binary': 'BYTEA',
        'image': 'BYTEA'
    }

    return type_mapping.get(data_type, 'TEXT')


def create_postgresql_table(target_engine, target_table, source_engine, source_table):
    """根據 SQL Server 結構自動創建 PostgreSQL 資料表"""
    try:
        # 解析目標表的 schema 和 table 名稱
        if '.' in target_table:
            target_schema, target_table_name = target_table.split('.', 1)
        else:
            target_schema = 'public'
            target_table_name = target_table

        # 擷取 SQL Server 資料表結構
        columns = get_sql_server_table_schema(source_engine, source_table)

        # 建立 PostgreSQL CREATE TABLE 語句
        column_definitions = []
        for col in columns:
            column_name = col[0]
            data_type = col[1]
            max_length = col[2]
            is_nullable = col[3]
            precision = col[4]
            scale = col[5]

            # 轉換資料類型
            pg_type = convert_sql_server_to_postgresql_type(data_type, max_length, precision, scale)

            # 處理 NULL 約束
            nullable = "" if is_nullable == 'YES' else "NOT NULL"

            column_def = f"{column_name} {pg_type} {nullable}".strip()
            column_definitions.append(column_def)

        # 建立完整的 CREATE TABLE 語句
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {target_schema}.{target_table_name} (
            {','.join(column_definitions)}
        );
        """

        logger.info(f"準備創建 PostgreSQL 資料表: {target_table}")
        logger.info(f"CREATE TABLE 語句: {create_table_sql}")

        with target_engine.begin() as conn:
            conn.execute(text(create_table_sql))

        logger.info(f"PostgreSQL 資料表 {target_table} 創建/確認完成")

    except Exception as e:
        logger.error(f"創建 PostgreSQL 資料表失敗: {e}")
        raise


def convert_dataframe_types(df):
    """轉換 DataFrame 資料類型以符合 PostgreSQL，將空白字串轉為 NULL"""
    try:
        # 處理所有欄位
        for col in df.columns:
            # 將空白字串轉為 None (NULL)
            df[col] = df[col].replace('', None)
            df[col] = df[col].replace(' ', None)  # 處理只有空格的情況

            # 如果是 object 類型且內容是字串，進一步清理
            if df[col].dtype == 'object':
                # 移除字串前後空白，如果結果是空字串則轉為 None
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('', None)
                df[col] = df[col].replace('nan', None)  # 處理字串化的 NaN
                df[col] = df[col].replace('None', None)  # 處理字串化的 None

        # 處理日期時間欄位
        datetime_columns = []
        for col in df.columns:
            # 檢查欄位名稱是否包含時間相關關鍵字
            if any(keyword in col.lower() for keyword in ['time', 'date', 'created', 'updated']):
                datetime_columns.append(col)

        for col in datetime_columns:
            if col in df.columns:
                # 確保日期時間格式正確，無效日期保持為 None
                df[col] = pd.to_datetime(df[col], errors='coerce')

        logger.info("資料類型轉換完成")
        logger.info(f"轉換後的資料類型: {df.dtypes.to_dict()}")
        logger.info(f"識別的日期時間欄位: {datetime_columns}")

        return df
    except Exception as e:
        logger.error(f"資料類型轉換失敗: {e}")
        raise


def sync_data_in_batches(source_engine, target_engine, source_table, target_table, total_rows):
    """分批同步資料"""
    try:
        offset = 0
        batch_size = SYNC_CONFIG['batch_size']
        chunksize = SYNC_CONFIG['chunksize']
        total_synced = 0

        # 解析目標表的 schema 和 table 名稱
        if '.' in target_table:
            target_schema, target_table_name = target_table.split('.', 1)
        else:
            target_schema = None
            target_table_name = target_table

        while offset < total_rows:
            logger.info(f"處理第 {offset + 1} 到 {min(offset + batch_size, total_rows)} 筆資料...")

            # 從 SQL Server 讀取資料
            query = f"""
            SELECT * FROM {source_table} 
            ORDER BY (SELECT NULL) 
            OFFSET {offset} ROWS 
            FETCH NEXT {batch_size} ROWS ONLY
            """

            df = pd.read_sql(query, source_engine)

            if df.empty:
                break

            # 轉換資料類型
            df = convert_dataframe_types(df)

            # 寫入 PostgreSQL
            try:
                with target_engine.begin() as conn:
                    df.to_sql(
                        name=target_table_name,
                        con=conn,
                        schema=target_schema,
                        if_exists='append',
                        index=False,
                        method='multi',
                        chunksize=chunksize
                    )
                logger.info(f"成功寫入 {len(df)} 筆資料")
            except Exception as e:
                logger.error(f"寫入失敗: {e}")
                logger.error(f"DataFrame shape: {df.shape}")
                logger.error(f"DataFrame dtypes: {df.dtypes}")
                # 顯示前幾筆資料以便除錯
                logger.error(f"DataFrame sample: {df.head(3).to_dict()}")
                raise

            total_synced += len(df)
            progress = (total_synced / total_rows) * 100
            logger.info(f"已同步 {total_synced:,} / {total_rows:,} 筆資料 ({progress:.1f}%)")

            offset += batch_size

        return total_synced
    except Exception as e:
        logger.error(f"資料同步失敗: {e}")
        raise


def verify_sync_result(target_engine, target_table, expected_count):
    """驗證同步結果"""
    try:
        query = f"SELECT COUNT(*) as count FROM {target_table}"
        with target_engine.connect() as conn:
            result = conn.execute(text(query))
            actual_count = result.fetchone()[0]

        if actual_count == expected_count:
            logger.info(f"同步驗證成功: 目標資料表共有 {actual_count:,} 筆資料")
            return True
        else:
            logger.error(f"同步驗證失敗: 預期 {expected_count:,} 筆，實際 {actual_count:,} 筆")
            return False
    except Exception as e:
        logger.error(f"同步驗證失敗: {e}")
        return False


def refresh_materialized_view(target_engine, mv_name):
    """更新實例化視圖"""
    try:
        logger.info(f"開始更新實例化視圖: {mv_name}")

        # 使用 CONCURRENTLY 選項進行並行更新（如果 MV 有唯一索引）
        # 如果沒有唯一索引，則使用一般更新
        refresh_query = f"REFRESH MATERIALIZED VIEW {mv_name}"

        try:
            with target_engine.begin() as conn:
                conn.execute(text(refresh_query))
            logger.info(f"實例化視圖 {mv_name} 更新成功 (CONCURRENTLY)")
        except Exception as e:
            # 如果 CONCURRENTLY 失敗（通常是因為沒有唯一索引），使用一般更新
            logger.warning(f"CONCURRENTLY 更新失敗，嘗試一般更新: {e}")

            refresh_query = f"REFRESH MATERIALIZED VIEW {mv_name}"
            with target_engine.begin() as conn:
                conn.execute(text(refresh_query))
            logger.info(f"實例化視圖 {mv_name} 更新成功")

    except Exception as e:
        logger.error(f"更新實例化視圖 {mv_name} 失敗: {e}")
        raise

# =============== 主程式 ===============
def sync_data_once():
    """執行一次資料同步"""
    start_time = datetime.now()
    logger.info(r"""
    ╔════════════════════════════════════╗
    ║          開始資料同步程序           
    ╚════════════════════════════════════╝
    """)

    logger.info(f"來源資料表: {SYNC_CONFIG['source_table']}")
    logger.info(f"目標資料表: {SYNC_CONFIG['target_table']}")

    source_engine = None
    target_engine = None

    try:
        # 建立連線
        target_engine = create_postgresql_connection()
        source_engine = create_sql_server_connection()

        # 創建 PostgreSQL 目標資料表
        create_postgresql_table(target_engine, SYNC_CONFIG['target_table'], source_engine, SYNC_CONFIG['source_table'])

        # 取得來源資料筆數
        total_rows = get_table_row_count(source_engine, SYNC_CONFIG['source_table'])

        # 清空目標資料表
        truncate_target_table(target_engine, SYNC_CONFIG['target_table'])

        # 同步資料
        synced_count = sync_data_in_batches(
            source_engine,
            target_engine,
            SYNC_CONFIG['source_table'],
            SYNC_CONFIG['target_table'],
            total_rows
        )

        # 驗證同步結果
        if verify_sync_result(target_engine, SYNC_CONFIG['target_table'], total_rows):
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"=== 資料同步完成 ===")
            logger.info(f"同步筆數: {synced_count:,}")
            logger.info(f"執行時間: {duration}")

            try:
                logger.info("=== 開始更新實例化視圖 ===")
                refresh_materialized_view(target_engine, "public.hd_mv_production_oee")
                logger.info("=== 實例化視圖更新完成 ===")
            except Exception as mv_error:
                logger.error(f"實例化視圖更新失敗，但資料同步已完成: {mv_error}")
                # 注意：這裡不返回 False，因為資料同步本身是成功的
                # 只是 MV 更新失敗，這樣可以讓程式繼續運行

            return True
        else:
            logger.error("資料同步驗證失敗")
            return False

    except Exception as e:
        logger.error(f"資料同步程序發生錯誤: {e}")
        return False
    finally:
        # 關閉連線
        if source_engine:
            source_engine.dispose()
            logger.info("SQL Server 連線已關閉")
        if target_engine:
            target_engine.dispose()
            logger.info("PostgreSQL 連線已關閉")


def main():
    """主要執行函數"""
    killer = GracefulKiller()
    schedule_interval = EXECUTION_CONFIG['schedule_interval']
    enable_schedule = EXECUTION_CONFIG['enable_schedule']

    logger.info("=== 程式啟動 ===")

    if enable_schedule and schedule_interval > 0:
        logger.info(f"定期執行模式: 每 {schedule_interval} 秒執行一次")
    else:
        logger.info("單次執行模式")

    execution_count = 0

    try:
        while True:
            execution_count += 1

            # 檢查是否需要停止
            if killer.kill_now:
                logger.info("接收到停止信號，程式即將退出")
                break

            logger.info(f"--- 第 {execution_count} 次執行開始 ---")
            logger.info(rf"""
            ======================
            ╭────────────────────╮
            │  第 {execution_count} 次執行開始
            ╰────────────────────╯
            ======================
            """)

            try:
                success = sync_data_once()

                if success:
                    logger.info(f"--- 第 {execution_count} 次執行成功 ---")
                else:
                    logger.error(f"--- 第 {execution_count} 次執行失敗 ---")
                    return False

            except Exception as e:
                logger.error(f"--- 第 {execution_count} 次執行失敗: {e} ---")
                return False

            # 如果不是定期執行模式，執行一次後退出
            if not enable_schedule or schedule_interval <= 0:
                break

            # 等待下次執行
            logger.info(f"等待 {schedule_interval} 秒後進行下次同步...")
            for i in range(schedule_interval):
                if killer.kill_now:
                    logger.info("接收到停止信號，程式即將退出")
                    return True
                time.sleep(1)

    except KeyboardInterrupt:
        logger.info("接收到中斷信號，程式即將退出")
        return True
    except Exception as e:
        logger.error(f"程式執行過程中發生未預期的錯誤: {e}")
        return False

    logger.info("=== 程式結束 ===")
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("程式執行成功")
            sys.exit(0)
        else:
            logger.error("程式執行失敗")
            sys.exit(1)
    except ValueError as e:
        logger.error(f"環境變數設定錯誤: {e}")
        logger.error("請確認以下環境變數已正確設定:")
        logger.error(
            "SQL Server: SQL_SERVER_HOST, SQL_SERVER_PORT, SQL_SERVER_DATABASE, SQL_SERVER_USERNAME, SQL_SERVER_PASSWORD")
        logger.error(
            "PostgreSQL: POSTGRESQL_HOST, POSTGRESQL_PORT, POSTGRESQL_DATABASE, POSTGRESQL_USERNAME, POSTGRESQL_PASSWORD")
        logger.error("同步設定: SYNC_SOURCE_TABLE, SYNC_TARGET_TABLE, SYNC_CHUNKSIZE, SYNC_BATCH_SIZE")
        logger.error("執行設定: EXECUTION_SCHEDULE_INTERVAL, EXECUTION_ENABLE_SCHEDULE")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程式啟動失敗: {e}")
        sys.exit(1)