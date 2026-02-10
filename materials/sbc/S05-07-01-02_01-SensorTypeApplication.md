---
layout: page
title:  "센서의 종류 및 응용 사례"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S05-07-01-02_01-SensorTypeApplication
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


- **센서의 종류**

<div class="info-table">
    <table>
        <thead>
            <th>센서 구분</th>
            <th colspan="2">설명</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader" rowspan="2">기계적 센서<br>(Mechanical Sensor)</td>
                <td style="width: 60px;">특징</td>
                <td class="td-left">물체의 위치, 변위, 압력, 가속도 등 기계적 상태 변화를 감지</td>
            </tr>
            <tr>
                <td>종류</td>
                <td class="td-left">초음파 센서, 압력 센서, 로드셀, 가속도계, 자이로스코프, 리미트 스위치, 인코더, 변위 센서, 진동 센서, 토크 센서 등</td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="2">전자기적 센서<br>(Electromagnetic Sensor)</td>
                <td>특징</td>
                <td class="td-left">
                    - 자기장의 세기나 전기적 특성 변화를 이용해 물체를 감지<br>
                    - 전선을 따라서 흐르는 전하의 흐름을 감지하여 전류가 얼마나 흐르는지 측정(유도전류 이용)
                </td>
            </tr>
            <tr>
                <td>종류</td>
                <td class="td-left">홀 센서, 자기 센서, 유도형 근접 센서, 전류 센서, 전압 센서, 금속 탐지 센서, 지자기 센서, 안테나 센서, 리드 스위치, 용량형 센서 등</td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="2">광학적 센서<br>(Optical Sensor)</td>
                <td>특징</td>
                <td class="td-left">빛의 반사, 투과, 굴절 등을 이용해 정보를 획득</td>
            </tr>
            <tr>
                <td>종류</td>
                <td class="td-left">포토다이오드, 조도 센서, 컬러 센서, 적외선 센서, 이미지 센서(CMOS/CCD), 레이저 거리 센서, 광전 스위치, UV 센서, 광섬유 센서, LiDAR 등</td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="2">방사선 센서<br>(Radiation Sensor)</td>
                <td>특징</td>
                <td class="td-left">
                    - X선, 감마선 등 방사성 물질이 방출하는 에너지를 측정<br>
                    - X-Ray: 예전에는 필름을 사용. 지금은 가시광 변환 패널 사용(신틸레이터->CCD칩)</td>
            </tr>
            <tr>
                <td>종류</td>
                <td class="td-left">가이거 계수기, 신틸레이션 검출기, 반도체 방사선 검출기, 중성자 검출기, X선 이미지 센서, 감마선 분광기, 라돈 센서 등</td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="2">음향 센서<br>(Acoustic Sensor)</td>
                <td>특징</td>
                <td class="td-left">
                    - 공기나 물속의 진동인 소리 에너지를 전기 신호로 변환<br>
                    - 마이크 --> 음파에 따라 프레임이 떨리고 떨림에 의해 진동, 전극변화 등을 이용해서 전기신호 생성
                </td>
            </tr>
            <tr>
                <td>종류</td>
                <td class="td-left">콘덴서 마이크, MEMS 마이크, 하이드로폰(수중 마이크), 초음파 수신 센서, 음압 센서, 소음계, 골전도 센서, 음향 방출(AE) 센서, 초음파 도플러 센서 등</td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="2">열 센서<br>(Thermal Sensor)</td>
                <td>특징</td>
                <td class="td-left">대상물이나 환경의 온도 변화 및 열에너지를 감지</td>
            </tr>
            <tr>
                <td>종류</td>
                <td class="td-left">서미스터(NTC/PTC), 열전대(Thermocouple), 백금 저항 온도계(RTD), 비접촉 적외선 온도 센서, 비접촉 열화상 센서, 바이메탈, 열류 센서 등</td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="2">화학 센서<br>(Chemical Sensor)</td>
                <td>특징</td>
                <td class="td-left">특정 가스나 액체 속 화학 물질의 성분 및 농도를 감지</td>
            </tr>
            <tr>
                <td>종류</td>
                <td class="td-left">CO 센서, CO2 센서, pH 센서, 전기화학식 가스 센서, 음주 측정 센서, 습도 센서, 연기 감지기, 반도체식 가스 센서, 암모니아 센서 등</td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="2">바이오 센서<br>(Bio Sensor)</td>
                <td>특징</td>
                <td class="td-left">
                    - 화학 센서의 일종<br>
                    - 생물학적 요소(효소, 항체 등)를 이용해 특정 물질을 선택적으로 분석
                </td>
            </tr>
            <tr>
                <td>종류</td>
                <td class="td-left">혈당 센서, 유전자(DNA) 센서, 항원-항체 반응 센서, 뇌파(EEG) 센서, 심전도(ECG) 센서, 근전도(EMG) 센서 등</td>
            </tr>
        </tbody>
    </table>
</div>