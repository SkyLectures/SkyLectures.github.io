---
layout: page
title:  "MongoDB 활용 및 실습"
date:   2026-05-31 09:00:00 +0900
permalink: /materials/S02-03-06-03_02-MongoDbPractice
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. MongoDB 기본 사용법

- MongoDB의 핵심 CRUD(생성, 조회, 수정, 삭제) 작업을 다루는 기본 사용법
- 코드는 가장 대중적으로 사용되는 백엔드 언어인 **Node.js(JavaScript)** 환경을 기준으로 작성함
- Node.js에서 공식 드라이버인 `mongodb` 패키지를 사용함

- **준비 작업 (라이브러리 설치)**
    - 터미널에서 MongoDB 드라이버를 프로젝트에 설치

        ```bash
        npm install mongodb
        ```

- **기본 CRUD 예제 코드**

    ```javascript
    //#file: "basic_curd.js"
    const { MongoClient, ObjectId } = require('mongodb');

    // MongoDB 연결 URI (로컬 환경 기준)
    const uri = "mongodb://localhost:27017";
    const client = new MongoClient(uri);

    async function run() {
        try {
            // 1. 데이터베이스 및 컬렉션 연결
            const database = client.db('shop_db');
            const products = database.collection('products');

            // [Create] 데이터 삽입 (insertOne, insertMany)
            const newProduct = { name: "무선 마우스", price: 35000, category: "전자제품" };
            const insertResult = await products.insertOne(newProduct);
            console.log(`[Create] 데이터 삽입 완료! ID: ${insertResult.insertedId}`);

            // [Read] 데이터 조회 (find, findOne)
            const query = { name: "무선 마우스" };
            const product = await products.findOne(query);
            console.log("[Read] 조회 결과:", product);

            // [Update] 데이터 수정 (updateOne, updateMany)
            const filter = { _id: product._id };
            const updateDoc = { $set: { price: 32000 } }; // 가격 할인
            const updateResult = await products.updateOne(filter, updateDoc);
            console.log(`[Update] ${updateResult.modifiedCount}개의 문서 수정됨.`);

            // [Delete] 데이터 삭제 (deleteOne)
            const deleteResult = await products.deleteOne({ _id: product._id });
            console.log(`[Delete] ${deleteResult.deletedCount}개의 문서 삭제됨.`);

        } finally {
            // 연결 종료
            await client.close();
        }
    }
    run().catch(console.dir);
    ```


## 2. MongoDB를 활용한 비정형 로그 데이터 적재

- **로그 데이터**
    - 서비스마다 포맷이 다르고 시간이 지나면서 필드가 동적으로 추가되기도 하는 대표적인 **비정형/반정형 데이터**
    - MongoDB는 고정된 스키마가 없기 때문에 이러한 대량의 로그 데이터를 적재하는 데 가장 이상적임

- **로그 적재 설계 포인트**

    1. **대량 적재(Bulk Write)**
        - 로그는 초당 수백~수천 건이 발생하므로
        - 개별 저장보다 한 번에 모아서 쏘는 `insertMany`를 사용하는 것이 성능상 유리함
    2. **시계열 데이터 최적화**
        - 최신 로그 조회가 잦으므로 시간(`timestamp`) 필드에 인덱스를 걸어줌
    3. **가변적 데이터 구조**
        - 서비스별 전용 데이터(에러 코드, 유저 디바이스 정보 등)를 `metadata`라는 서브 도큐먼트에 유연하게 담아냄

- 실습 예제

    - 예제 실행을 위한 기본 데이터 및 생성 스크립트
        - 테스트를 진행할 수 있도록 서버에서 실시간으로 발생하는 것처럼 꾸며진 **비정형 로그 데이터셋 자동 생성 프로그램**
        - 파일로 다운로드할 필요 없이 아래 코드를 실행하면 가상의 비정형 데이터가 자동으로 적재됨

        - **로그 적재 및 대량 생성 전체 코드 (`log_ingestor.js`)**

            ```javascript
            //#file: "log_ingestor.js"
            const { MongoClient } = require('mongodb');

            const uri = "mongodb://localhost:27017";
            const client = new MongoClient(uri);

            // 1. 테스트를 위한 비정형 가상 로그 데이터 생성기
            function generateMockLogs() {
                const services = ['auth-service', 'payment-service', 'order-service', 'delivery-module'];
                const levels = ['INFO', 'WARN', 'ERROR'];
                const mockLogs = [];

                // 서로 다른 구조를 가진 비정형 로그 3가지를 무작위로 생성합니다.
                for (let i = 0; i < 50; i++) {
                    const service = services[Math.floor(Math.random() * services.length)];
                    const level = levels[Math.floor(Math.random() * levels.length)];
                    
                    let baseLog = {
                        timestamp: new Date(Date.now() - Math.random() * 10000000), // 무작위 과거 시간
                        level: level,
                        service: service
                    };

                    // 조건에 따라 내부 데이터 구조(Schema)가 완전히 달라지는 비정형 특성 반영
                    if (level === 'ERROR') {
                        baseLog.message = "System connection failed.";
                        baseLog.error_details = {
                            code: 500 + Math.floor(Math.random() * 5),
                            trace_id: `err_${Math.random().toString(36).substr(2, 9)}`,
                            retry_count: Math.floor(Math.random() * 3)
                        };
                    } else if (level === 'WARN') {
                        baseLog.message = "Resource threshold warning.";
                        baseLog.metrics = {
                            cpu_usage: parseFloat((80 + Math.random() * 15).toFixed(2)),
                            memory_mb: 4096
                        };
                    } else {
                        baseLog.message = "User activity recorded.";
                        baseLog.user_activity = {
                            user_id: `user_${Math.floor(Math.random() * 1000)}`,
                            action: ['click_banner', 'view_item', 'add_to_cart'][Math.floor(Math.random() * 3)],
                            ip: "192.168.1.100"
                        };
                    }

                    mockLogs.push(baseLog);
                }
                return mockLogs;
            }

            async function main() {
                try {
                    const database = client.db('logging_system');
                    const logsCollection = database.collection('server_logs');

                    console.log("🛠️ 1. 비정형 가상 로그 데이터 50건 생성 중...");
                    const dummyLogs = generateMockLogs();

                    console.log("🚀 2. MongoDB에 비정형 로그 대량 적재(Bulk Insert) 시작...");
                    // insertMany를 이용해 대량의 비정형 데이터를 효율적으로 한 번에 삽입
                    const result = await logsCollection.insertMany(dummyLogs);
                    console.log(`✅ 적재 완료! 총 ${result.insertedCount}개의 로그가 저장되었습니다.`);

                    console.log("🔍 3. 특정 조건(에러 로그 중 고유 코드 보유) 검색 테스트...");
                    // 구조가 다른 데이터 중 'error_details' 필드가 존재하는 에러 로그만 뽑아내기
                    const errorQuery = { level: "ERROR", "error_details.code": { $exists: true } };
                    const errorLogs = await logsCollection.find(errorQuery).limit(2).toArray();
                    
                    console.log("=== 에러 로그 검색 결과 샘플 ===");
                    console.dir(errorLogs, { depth: null });

                    console.log("⚡ 4. 빠른 로그 조회를 위한 시계열 인덱스 생성...");
                    await logsCollection.createIndex({ timestamp: -1 });
                    await logsCollection.createIndex({ level: 1, timestamp: -1 }); // 복합 인덱스
                    console.log("✅ 인덱스 생성 완료.");

                } catch (error) {
                    console.error("오류 발생:", error);
                } finally {
                    await client.close();
                }
            }

            main();
            ```

    - **실행 방법**

        1. 로컬에 MongoDB가 켜져 있는지 확인 (포트 `27017`)
        2. 프로젝트 디렉토리에 위 코드를 `log_ingestor.js`로 저장
        3. 터미널에 `node log_ingestor.js`를 입력해 실행
        4. 실행 결과 콘솔창에 서로 다른 필드 구조(`error_details`, `metrics`, `user_activity`)를 가진 데이터가 성공적으로 한 곳에 적재되고 조회되는 모습을 확인