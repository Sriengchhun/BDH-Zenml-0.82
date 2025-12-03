import os
import psycopg2
import json
import pytz

from psycopg2 import sql
from datetime import datetime

DATABASE_NAME = str(os.environ.get("POSTGRES_DATABASE_NAME"))
USER_NAME = str(os.environ.get("POSTGRES_USER_NAME"))
PASSWORD = str(os.environ.get("POSTGRES_PASSWORD"))
HOST_IP = str(os.environ.get("POSTGRES_HOST_IP"))
PORT = str(os.environ.get("POSTGRES_PORT"))

# Configure postgreSQL connection
db_config = {
    "dbname": DATABASE_NAME,
    "user": USER_NAME,
    "password": PASSWORD,
    "host": HOST_IP,
    "port": PORT
}

class ConnectPostgresqlDB():
    def __init__(self) -> None:
        pass

    '''create tabel, insert data and update data from training '''

    def check_and_create_table(self):
        # Script SQL for check table
        check_table_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'model'
            );
        """
        
        # Script SQL for created table
        create_table_query = """
            CREATE TABLE Model (
                model_id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                project_id VARCHAR(255) NOT NULL,
                category VARCHAR(100) NOT NULL,
                model_name VARCHAR(255) NOT NULL,
                uuid_name VARCHAR(255),
                uuid VARCHAR(255) NOT NULL,
                parameter JSONB,
                status JSONB,
                queue VARCHAR(50),
                graph JSONB,
                evaluation JSONB,
                pipeline_name VARCHAR(255),
                run_name VARCHAR(255),
                version VARCHAR(255),
                service_name VARCHAR(255),
                service_version VARCHAR(255),
                uuid_deployer VARCHAR(255),
                status_deployer TEXT,
                port VARCHAR(50),
                created_at TIMESTAMP,
                last_update TIMESTAMP
            );
        """
        
        try:
            # เชื่อมต่อกับฐานข้อมูล
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            
            # เช็คว่ามีตารางหรือไม่
            cursor.execute(check_table_query)
            table_exists = cursor.fetchone()[0]
            
            # ถ้าไม่มีตาราง ให้สร้างตาราง
            if not table_exists:
                cursor.execute(create_table_query)
                conn.commit()
                print("Table 'Model' has been created.")

                cursor.close()
                conn.close()
            else:
                print("Table 'Model' already exists.")
        
        except Exception as e:
            print(f"An error occurred: {e}")
    

    def insert_data(self, user_id, model_name, uuid_name, project_id, category, uuid, parameter, status, queue):
        # self.check_and_create_table()

        # set timestamp format
        time_zone = pytz.timezone('Asia/Bangkok')
        current_datetime = datetime.now(time_zone)
        timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
    

        query_insert ="""
                INSERT INTO model 
                (
                    user_id, model_name, uuid_name, project_id, category, uuid, 
                    parameter, status, queue, created_at, last_update
                )
                VALUES((%s), (%s), (%s), (%s), (%s), (%s), (%s), (%s), (%s), (%s), (%s))
            """

        cursor.execute(query_insert, 
            (user_id, model_name, uuid_name, project_id, category, uuid, parameter, status, queue, timestamp, timestamp)
        )

        conn.commit()
        cursor.close()
        conn.close()

    def update_status(self, model_id, status):
        # self.check_and_create_table()

        # set timestamp format
        time_zone = pytz.timezone('Asia/Bangkok')
        current_datetime = datetime.now(time_zone)
        timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        query ="""
                UPDATE model SET status = %s, last_update = %s 
                WHERE model_id = %s 
            """
        cursor.execute(query, (status, timestamp, model_id))

        conn.commit()
        cursor.close()
        conn.close()


    def update_run_name(self, model_id, run_name):
        # self.check_and_create_table()

        # set timestamp format
        time_zone = pytz.timezone('Asia/Bangkok')
        current_datetime = datetime.now(time_zone)
        timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        query ="""
                UPDATE model SET run_name = %s, last_update = %s 
                WHERE model_id = %s 
            """
        cursor.execute(query, (run_name, timestamp, model_id))

        conn.commit()
        cursor.close()
        conn.close()



    def update_model(self, model_id, version, service_name, service_version):
        # self.check_and_create_table()

        # set timestamp format
        time_zone = pytz.timezone('Asia/Bangkok')
        current_datetime = datetime.now(time_zone)
        timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        query ="""
                UPDATE model SET version = %s, service_name = %s, service_version = %s, last_update = %s 
                WHERE model_id = %s 
            """
        cursor.execute(query, (version, service_name, service_version, timestamp, model_id))

        conn.commit()
        cursor.close()
        conn.close()


    def update_deployer(self, model_id, pipeline_name, run_name, uuid_deployer, status_deployer, port):
        # self.check_and_create_table()

        # set timestamp format
        time_zone = pytz.timezone('Asia/Bangkok')
        current_datetime = datetime.now(time_zone)
        timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        query_insert ="""
                UPDATE model SET pipeline_name = %s, run_name = %s, uuid_deployer = %s, status_deployer = %s, port = %s, 
                    last_update = %s 
                WHERE model_id = %s 
            """
        cursor.execute(query_insert, (pipeline_name, run_name, uuid_deployer, status_deployer, port, timestamp, model_id))

        conn.commit()
        cursor.close()
        conn.close()


    def get_designated_queue_number(self):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT queue FROM model 
                WHERE queue not in ('completed', 'error', 'delete')
                ORDER BY CAST(queue AS INT) desc limit 1
            """
        cursor.execute(query)

        queue_number = 1

        if cursor.rowcount != 0:
            rows = cursor.fetchall()
            queue_number = str(int(rows[0][0]) + 1)

        cursor.close()
        conn.close()

        return queue_number


    def get_current_tasks(self):
        # self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT * FROM model 
                WHERE ((status->>'task_status' = '1') OR NOT status ? 'task_status') 
	                and queue not in ('completed', 'error', 'delete')
                order by last_update desc
            """
        cursor.execute(query)

        data = []

        if cursor.rowcount != 0:
            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data.append(record)

        cursor.close()
        conn.close()

        return data    


    def get_queue_tasks(self):
        # self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT model_id, uuid_name, category, parameter FROM model 
                WHERE queue not in ('completed', 'error', 'delete') and status->>'task_status' = '0'
                order by CAST(queue AS INT) asc limit 1
            """
        cursor.execute(query)

        data = []

        if cursor.rowcount != 0:
            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data = record

        cursor.close()
        conn.close()

        return data  


    def wait_in_queue(self, model_id):
        status = json.dumps({ "output": "Wait for queue", "success": True, "task_status": 0 })
        self.update_status(model_id, status)


    def queue_on_process(self, model_id):
        status = json.dumps({ "output": "Processing started", "success": True, "task_status": 1 })
        self.update_status(model_id, status)


    def queue_on_train(self, model_id):
        status = json.dumps({ "output": "Training started", "success": True, "task_status": 1 })
        self.update_status(model_id, status)


    def queue_on_deployment(self, model_id):
        status = json.dumps({ "output": "Deployment started", "success": True, "task_status": 1 })
        self.update_status(model_id, status)


    def update_queue(self, model_id, queue):
        # self.check_and_create_table()

        # set timestamp format
        time_zone = pytz.timezone('Asia/Bangkok')
        current_datetime = datetime.now(time_zone)
        timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                UPDATE model SET queue = %s, last_update = %s WHERE model_id = %s 
            """
        cursor.execute(query, (queue, timestamp, model_id))

        conn.commit()
        cursor.close()
        conn.close()


    def queue_complete(self, model_id):
        self.update_queue(model_id=model_id, queue="completed")

    def queue_error(self, model_id):
        self.update_queue(model_id=model_id, queue="error")

    def queue_delete(self, model_id):
        self.update_queue(model_id=model_id, queue="delete")


    def check_queue(self):
        # self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT model_id, queue FROM model 
                WHERE queue not in ('completed', 'error', 'delete') and status->>'task_status' = '1'
                order by CAST(queue AS INT) asc
            """
        cursor.execute(query)

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data.append(record)

        cursor.close()
        conn.close()

        return data


    def inspect_queued_tasks(self):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT model_id, uuid_name, status, run_name, status_deployer FROM model 
                WHERE queue not in ('completed', 'error', 'delete')
                order by CAST(queue AS INT) ASC limit 1
            """
        cursor.execute(query)

        data = []

        if cursor.rowcount != 0:
            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data = record

        cursor.close()
        conn.close()

        return data

    ''' get data from db'''

    def datetime_serializer(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        raise TypeError("Type not serializable")


    def get_by_id(self, model_id):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT model_id, user_id, project_id, model_name, parameter, uuid_name, uuid, status, queue, 
                version, pipeline_name, uuid_deployer, status_deployer, port, created_at, last_update
                FROM model WHERE model_id = %s
                order by last_update desc
            """
        cursor.execute(query, (model_id,))

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data.append(record)

        cursor.close()
        conn.close()

        # Convert Dict to JSON format
        j = json.dumps(data, ensure_ascii=False, default=self.datetime_serializer).encode('utf8')

        return j


    def get_by_uuid(self, user_id, uuid):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT model_id, user_id, project_id, model_name, parameter, uuid_name, uuid, status, queue, 
                version, pipeline_name, uuid_deployer, status_deployer, port, created_at, last_update
                FROM model WHERE user_id = %s and uuid = %s
                order by last_update desc
            """
        cursor.execute(query, (user_id, uuid))

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data.append(record)

        cursor.close()
        conn.close()

        # Convert Dict to JSON format
        j = json.dumps(data, ensure_ascii=False, default=self.datetime_serializer).encode('utf8')

        return j


    def get_by_project(self, user_id, project_id):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT model_id, user_id, project_id, model_name, parameter, uuid_name, uuid, status, queue, 
                version, pipeline_name, uuid_deployer, status_deployer, port, created_at, last_update
                FROM model WHERE user_id = %s and project_id = %s
                order by last_update desc
            """
        cursor.execute(query, (user_id, project_id))

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data.append(record)

        cursor.close()
        conn.close()

        # Convert Dict to JSON format
        j = json.dumps(data, ensure_ascii=False, default=self.datetime_serializer).encode('utf8')

        return j


    def get_data_by_user(self, user_id):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT model_id, user_id, project_id, model_name, parameter, uuid_name, uuid, status, queue, 
                version, pipeline_name, uuid_deployer, status_deployer, port, created_at, last_update
                FROM model WHERE user_id = %s
                order by last_update desc
            """
        cursor.execute(query, (user_id,))

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data.append(record)

        cursor.close()
        conn.close()

        # Convert Dict to JSON format
        j = json.dumps(data, ensure_ascii=False, default=self.datetime_serializer).encode('utf8')

        return j


    def all_user_model_history(self):
        self.check_and_create_table()
        
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT model_id, user_id, project_id, model_name, parameter, uuid_name, uuid, status, queue, 
                version, pipeline_name, uuid_deployer, status_deployer, port, created_at, last_update
                From model
                order by last_update desc
            """
        cursor.execute(query)

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data.append(record)

        cursor.close()
        conn.close()

        # Convert Dict to JSON format
        j = json.dumps(data, ensure_ascii=False, default=self.datetime_serializer).encode('utf8')

        return j


    def search_model(self, model_id):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT project_id, category, model_name, uuid_name, version FROM model WHERE model_id = %s
            """
        cursor.execute(query, (model_id,))

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data = record

        cursor.close()
        conn.close()

        return data


    def get_uuid_deployer(self, model_id):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT uuid_name, uuid_deployer, category, parameter FROM model WHERE model_id = %s
            """
        cursor.execute(query, (model_id,))

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data = record

        cursor.close()
        conn.close()

        cursor.close()
        conn.close()

        return data


    def get_local_data(self, model_id):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT uuid_name, run_name, service_name, uuid_deployer
                FROM model WHERE model_id = %s
            """
        cursor.execute(query, (model_id,))

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data = record

        cursor.close()
        conn.close()

        return data
    

    def update_epoch(self, model_id, epoch):
        # self.check_and_create_table()

        # set timestamp format
        time_zone = pytz.timezone('Asia/Bangkok')
        current_datetime = datetime.now(time_zone)
        timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

        status = json.dumps({ "output": "Processing started", "success": True, "task_status": 1, "epochs": f"{epoch}" })

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                UPDATE model SET status = %s, last_update = %s WHERE model_id = %s 
            """
        cursor.execute(query, (status, timestamp, model_id))

        conn.commit()
        cursor.close()
        conn.close()
    

    def get_active_services(self):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT model_id FROM model where status_deployer  = '✅'
            """
        cursor.execute(query)

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data.append(record)
        
        cursor.close()
        conn.close()

        return data


    def reset_service_status(self, model_id):
        # set timestamp format
        time_zone = pytz.timezone('Asia/Bangkok')
        current_datetime = datetime.now(time_zone)
        timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                UPDATE model SET port = NULL, status_deployer = '⏸', last_update = %s 
                WHERE model_id = %s
            """
        cursor.execute(query, (timestamp, model_id))
        
        conn.commit()
        cursor.close()
        conn.close()


    def update_graph(self, model_id, graph):
        # self.check_and_create_table()

        time_zone = pytz.timezone('Asia/Bangkok')
        current_datetime = datetime.now(time_zone)
        timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        query_insert ="""
                UPDATE model SET graph = %s, last_update = %s 
                WHERE model_id = %s 
            """
        cursor.execute(query_insert, (graph , timestamp, model_id))

        conn.commit()
        cursor.close()
        conn.close()


    def update_evaluation(self, model_id, evaluate):
        # self.check_and_create_table()

        time_zone = pytz.timezone('Asia/Bangkok')
        current_datetime = datetime.now(time_zone)
        timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        query_insert ="""
                UPDATE model SET evaluation = %s, last_update = %s 
                WHERE model_id = %s 
            """
        cursor.execute(query_insert, (evaluate , timestamp, model_id))

        conn.commit()
        cursor.close()
        conn.close()


    def get_graph(self, model_id):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT graph 
                FROM model WHERE model_id = %s
            """
        cursor.execute(query, (model_id,))

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data = record

        cursor.close()
        conn.close()

        return data



    def get_evaluation(self, model_id):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT evaluation 
                FROM model WHERE model_id = %s
            """
        cursor.execute(query, (model_id,))

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data = record

        cursor.close()
        conn.close()

        return data


    def get_count_by_userID(self, user_id):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT COUNT(*) AS model_id 
                FROM model where user_id = %s and queue not in ('delete')
            """
        cursor.execute(query, (user_id,))

        data = 0

        if cursor.rowcount != 0:
            rows = cursor.fetchall()
            data = int(rows[0][0])

        cursor.close()
        conn.close()

        return data


    def query_user(self):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT user_id, COUNT(*) FROM model group by user_id
            """
        cursor.execute(query,)

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data.append(record)

        cursor.close()
        conn.close()

        # Convert Dict to JSON format
        j = json.dumps(data, ensure_ascii=False, default=self.datetime_serializer).encode('utf8')

        return j

    def query_model(self, project_id, model_type):
        self.check_and_create_table()

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        query ="""
                SELECT model_id, model_name FROM model WHERE project_id = %s and parameter ->> 'neural_network_architecture' = %s and queue not in ('error','delete')
            """
        cursor.execute(query, (project_id, model_type))

        data = []

        if cursor.rowcount != 0:

            keys = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                record = dict(zip(keys, row))
                data.append(record)

        cursor.close()
        conn.close()

        # Convert Dict to JSON format
        j = json.dumps(data, ensure_ascii=False, default=self.datetime_serializer).encode('utf8')

        return j
