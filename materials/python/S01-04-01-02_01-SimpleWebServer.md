---
layout: page
title:  "Simple 웹 서버 개발 예제"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-01-02_01-SimpleWebServer
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 웹서버 로그 처리하기

```bash
!wget https://raw.githubusercontent.com/SkyLectures/LectureMaterials/main/datasets/S01-04-01-001_001-access_log -O ./access_log
```

### 1.1 총 페이지 뷰 수 계산하기

```python
pageviews = 0

with open('access_log', 'r') as f:
   logs = f.readlines()
   for log in logs:
      log = log.split()
      status = log[8]
      if status == '200':
         pageviews += 1

print('총 페이지뷰: [%d]' %pageviews)
```

### 1.2 고유 방문자 수 계산하기

```python
visit_ip = []

with open('access_log', 'r') as f:
   logs = f.readlines()
   for log in logs:
      log = log.split()
      ip = log[0]
      if ip not in visit_ip:
         visit_ip.append(ip)

print('고유 방문자수: [%d]' %len(visit_ip))
```

### 1.3 총 서비스 용량 계산하기

```python
KB = 1024
total_service = 0

with open('access_log', 'r') as f:
   logs = f.readlines()
   for log in logs:
      log = log.split()
      servicebyte = log[9]
      if servicebyte.isdigit():
         total_service += int(servicebyte)

total_service /= KB
print('총 서비스 용량: %dKB' %total_service)
```

### 1.4 사용자별 서비스 용량 계산하기

```python
services = {}

with open('access_log', 'r') as f:
   logs = f.readlines()
   for log in logs:
      log = log.split()
      ip = log[0]
      servicebyte = log[9]
      if servicebyte.isdigit():
         servicebyte = int(servicebyte)
      else:
         servicebyte = 0

      if ip not in services:
         services[ip] = servicebyte
      else:
         services[ip] += servicebyte

ret = sorted(services.items(), key=lambda x: x[1], reverse=True)

print('사용자IP – 서비스용량')
for ip, b in ret:
   print('[%s] – [%d]' %(ip, b))
```

## 2. 에코 서버 만들기(1)

- 에코 서버
    - 네트워크로 메시지를 수신하여 송신자에게 수신한 메시지를 귿로 돌려보내는 서버
    - 네트워크 통신의 기본이 되는 소켓 프로그래밍을 접해보자

- 네트워크 소켓
    - 네트워크 통신에 있어서 시작점이자 끝점
    - 클라이언트-서버 프로그램의 가장 핵심이 되는 모듈
    - 서버와 클라이언트는 각자의 네트워크 소켓을 가지고 있음
    - 네트워크 통신을 위해 사용되는 프로토콜의 종류에 따라 TCP/UDP/Raw 소켓으로 구분됨
        - TCP 소켓:
            - TCP(Transmission Control Protocol)를 활용하는 네트워크 소켓
            - TCP
                - 연결 지향적 프로토콜
                - 포트 번호를 이용하여 서비스를 식별
                - 데이터의 신뢰성 있는 전송을 보장함
                - 데이터의 순서, 무결성, 신뢰성을 보장함
                - 주로 웹, 메일, 파일 공유와 같이 데이터 누락을 허용하지 않는 서비스에서 사용됨
        - UDP 소켓:
            - UDP(User Datagram Protocol)를 활용하는 네트워크 소켓
            - UDP
                - 비연결 지향적 프로토콜
                - 포트 번호를 이용하여 서비스를 식별
                - 빠른 데이터 전송을 목적으로함
                - 연결 설정 과정 없이 데이터를 전송
                - 데이터의 순서나 무결성은 보장하지 않음
                - VoIP (Voice over IP)와 같이 속도가 중요한 서비스에서 사용됨
        - Raw 소켓:
            - 특정한 프로토콜에 대한 전송 계층 포매팅 없이 인터넷 프로토콜 패킷을 직접 주고 받을 수 있는 인터넷 소켓
            - 데이터를 전송할 때 직접 프로토콜 헤더를 만들어 전송하고, 데이터를 수신할 때도 프로토콜 헤더를 포함하여 수신
            - 라우터나 네트워크 장비 등에 주로 활용되며 사용자 공간에서 새로운 전송 계층 프로토콜을 구현하는 데에도 활용됨

<p align="center"><img src="/materials/images/python/S01-04-01-00_005.jpg" width="700"></p>

### 2.1 서버

```python
#//file: "echo_server1.py"
import socket

HOST = ''
PORT = 9009

def runServer():
   with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.bind((HOST, PORT))
      sock.listen(1)
      print('클라이언트 연결을 기다리는 중..')
      conn, addr = sock.accept()

      with conn:
         print('[%s]와 연결됨' %addr[0])
         while True:
            data = conn.recv(1024)
            if not data:
               break
            print('메시지 수신 [%s]' %data.decode())
            conn.sendall(data)

runServer()
```

### 2.2 클라이언트

```python
#//file: "echo_client1.py"
import socket

HOST = 'localhost'
PORT = 9009

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
   sock.connect((HOST, PORT))
   msg = input('메시지 입력: ')
   sock.sendall(msg.encode())
   data = sock.recv(1024)

print('에코 서버로부터 받은 데이터 [%s]' %data.decode())
```

## 3. 에코 서버 만들기(2)

### 3.1 서버

```python
#//file: "echo_server2.py"
import socketserver

HOST = ''
PORT = 9009

class MyTcpHandler(socketserver.BaseRequestHandler):
   # 이 클래스는 서버 하나당 단한번 초기화됩니다.
   # handle() 메쏘드에 클라이언트 연결 처리를 위한 로직을 구현합니다.
   def handle(self):
      print('[%s] 연결됨' %self.client_address[0])

      try:
         while True:
            self.data = self.request.recv(1024)
            if self.data.decode() == '/quit':
               print('[%s] 사용자에 의해 중단' %self.client_address[0])
               return

            print('[%s]' %self.data.decode())
            self.request.sendall(self.data)
      except Exception as e:
         print(e)

def runServer():
   print('+++ 에코 서버를 시작합니다.')
   print('+++ 에코 서버를 끝내려면 Ctrl-C를 누르세요.')

   try:
       server = socketserver.TCPServer((HOST, PORT), MyTcpHandler)
       server.serve_forever()
   except KeyboardInterrupt:
      print('--- 에코 서버를 종료합니다.')

runServer()
```

### 3.2 클라이언트

```python
#//file: "echo_client2.py"
import socket

HOST = 'localhost'
PORT = 9009

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
   sock.connect((HOST, PORT))

   while True:
       msg = input('메시지 입력: ')
       if msg == '/quit':
           sock.sendall(msg.encode())
           break

       sock.sendall(msg.encode())
       data = sock.recv(1024)
       print('에코 서버로부터 받은 데이터 [%s]' %data.decode())

print('클라이언트 종료')
```

## 4. 파일 전송 프로그램 만들기

### 4.1 파일 송신 프로그램

```python
#//file: "file_server.py"
import socketserver
from os.path import exists

HOST = ''
PORT = 9009

class MyTcpHandler(socketserver.BaseRequestHandler):
   def handle(self):
      data_transferred = 0
      print('[%s] 연결됨' %self.client_address[0])
      filename = self.request.recv(1024)
      filename = filename.decode()

      if not exists(filename):
         return

      print('파일 [%s] 전송 시작...' %filename)
      with open(filename, 'rb') as f:
         try:
            data = f.read(1024)
            while data:
               data_transferred += self.request.send(data)
               data = f.read(1024)
         except Exception as e:
            print(e)

      print('전송완료[%s], 전송량[%d]' %(filename, data_transferred))

def runServer():
   print('+++ 파일 서버를 시작합니다.')
   print('+++ 파일 서버를 끝내려면 Ctrl-C를 누르세요.')

   try:
      server = socketserver.TCPServer((HOST, PORT), MyTcpHandler)
      server.serve_forever()
   except KeyboardInterrupt:
      print('--- 파일 서버를 종료합니다.')

runServer()
```

### 4.2 파일 수신 프로그램

```python
#//file: "file_client.py"
import socket

HOST = 'localhost'
PORT = 9009

def getFileFromServer(filename):
   data_transferred = 0

   with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.connect((HOST, PORT))
      sock.sendall(filename.encode())

      data = sock.recv(1024)
      if not data:
         print('파일[%s]: 서버에 존재하지 않거나 전송중 오류발생' %filename)
         return

      with open('download/'+filename, 'wb') as f:
         try:
            while data:
               f.write(data)
               data_transferred += len(data)
               data = sock.recv(1024)
         except Exception as e:
            print(e)

   print('파일 [%s] 전송종료. 전송량 [%d]' %(filename, data_transferred))

filename = input('다운로드 받을 파일이름을 입력하세요: ')
getFileFromServer(filename)
```

## 5. 채팅 서비스 만들기

### 5.1 채팅 서버 만들기

```python
#//file: "chat_server.py"
import socketserver
import threading

HOST = ''
PORT = 9009
lock = threading.Lock()

class UserManager:
   def __init__(self):
      self.users = {}

   def addUser(self, username, conn, addr):
      if username in self.users:
         conn.send('이미 등록된 사용자입니다.\n'.encode())
         return None

      # 새로운 사용자를 등록함
      lock.acquire()
      self.users[username] = (conn, addr)
      lock.release()

      self.sendMessageToAll('[%s]님이 입장했습니다.' %username)
      print('+++ 대화 참여자 수 [%d]' %len(self.users))

      return username

   def removeUser(self, username):
      if username not in self.users:
         return

      lock.acquire()
      del self.users[username]
      lock.release()

      self.sendMessageToAll('[%s]님이 퇴장했습니다.' %username)
      print('--- 대화 참여자 수 [%d]' %len(self.users))

   def messageHandler(self, username, msg):
      if msg[0] != '/':
         self.sendMessageToAll('[%s] %s' %(username, msg))
         return

      if msg.strip() == '/quit':
         self.removeUser(username)
         return -1

   def sendMessageToAll(self, msg):
      for conn, addr in self.users.values():
         conn.send(msg.encode())

class MyTcpHandler(socketserver.BaseRequestHandler):
   userman = UserManager()

   def handle(self):
      print('[%s] 연결됨' %self.client_address[0])

      try:
         username = self.registerUsername()
         msg = self.request.recv(1024)
         while msg:
            print(msg.decode())
            if self.userman.messageHandler(username, msg.decode()) == -1:
               self.request.close()
               break
            msg = self.request.recv(1024)

      except Exception as e:
         print(e)

      print('[%s] 접속종료' %self.client_address[0])
      self.userman.removeUser(username)

   def registerUsername(self):
      while True:
         self.request.send('로그인ID:'.encode())
         username = self.request.recv(1024)
         username = username.decode().strip()
         if self.userman.addUser(username, self.request, self.client_address):
            return username

class ChatingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

def runServer():
   print('+++ 채팅 서버를 시작합니다.')
   print('+++ 채텅 서버를 끝내려면 Ctrl-C를 누르세요.')

   try:
      server = ChatingServer((HOST, PORT), MyTcpHandler)
      server.serve_forever()
   except KeyboardInterrupt:
      print('--- 채팅 서버를 종료합니다.')
      server.shutdown()
      server.server_close()

runServer()
```

### 5.2 채팅 클라이언트 만들기

```python
#//file: "chat_client.py"
import socket
from threading import Thread

HOST = 'localhost'
PORT = 9009

def rcvMsg(sock):
   while True:
      try:
         data = sock.recv(1024)
         if not data:
            break
         print(data.decode())
      except:
         pass

def runChat():
   with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.connect((HOST, PORT))
      t = Thread(target=rcvMsg, args=(sock,))
      t.daemon = True
      t.start()

      while True:
         msg = input()
         if msg == '/quit':
            sock.send(msg.encode())
            break

         sock.send(msg.encode())

runChat()
```