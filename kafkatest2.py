from kafka import KafkaProducer
from kafka import KafkaConsumer
import time

topic = 'foobar'
bootstrap_servers ='192.168.9.130:9092'



producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

for x in range(20):
    time.sleep(2)
    #future = producer.send('foobar', send_str)
    key = 'foo'+str(x)
    value = 'bar'+str(x)
    future = producer.send(topic, key=key.encode('utf-8'), value=value.encode('utf-8'))
    result = future.get(timeout=60)
    print(result)



'''
consumer = KafkaConsumer(topic,bootstrap_servers=bootstrap_servers, auto_offset_reset='latest')
for msg in consumer:
    key = msg.key.decode(encoding="utf-8")  # 因为接收到的数据时bytes类型，因此需要解码
    value = msg.value.decode(encoding="utf-8")
    print("%s-%d-%d key=%s value=%s" % (msg.topic, msg.partition, msg.offset, key, value))
'''
