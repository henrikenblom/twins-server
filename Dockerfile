FROM henrikenblom/face-recognition-server:latest

WORKDIR /app

ADD $PWD/*.py /app

EXPOSE 3001

CMD ["python3", "twins-server.py"]