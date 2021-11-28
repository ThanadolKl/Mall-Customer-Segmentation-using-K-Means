# Mall Customer Segmentation using K-Means
## Author : Thanadol Klainin 6S No.8
## ESC 782 DSS


## Data Cleaning

> Clean data โดยลบ column ที่อาจจะไม่จำเป็น และเป็น Binary categories เช่น column Gender (ซึ่งจริง ๆ ก็สามารถนำมาใช้ในการวิเคราห์ได้หากสินค้าเป็นสินค้าที่แบ่งแยกเพศ แต่ในที่นี้ขอไม่ใช้ column นี้)

~~~

customer_df = df.drop(columns=['CustomerID', 'Gender'])
~~~

## Standardization

> จะเห็นว่าข้อมูล df ใน column spending scores คือข้อมูลที่ Standardize มาแล้ว และหากดู Range ของข้อมูลแต่ละ column จะพบว่า Range ไม่ได้ต่างกันมาก จึงไม่ได้มีความจำเป็นที่ต้องทำการ Normalization, Standardize เพิ่มเติม 

## Silhouette Analysis

> ก่อนที่จะสร้าง model ด้วยจำนวน n_clusters ที่ต้องกำหนดขึ้นมาเอง ผมลองหาจำนวน n_clusters ที่เหมาะสมผ่านการคิด Silhouette scores และ visualize ออกมาเป็นกราฟได้ดังนี้

~~~
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer ## สำหรับการ Visualize 
###
for n_cluster in [2,3,4,5,6,7,8]:
    kmeans = KMeans(n_clusters=n_cluster).fit(
        customer_df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    silhouette_avg = silhouette_score(
        customer_df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], 
        kmeans.labels_)
    
    print('Silhouette Score for %i Clusters: %0.4f' % (n_cluster, silhouette_avg))
~~~    

### Output

~~~
Silhouette Score for 2 Clusters: 0.2932
Silhouette Score for 3 Clusters: 0.3839
Silhouette Score for 4 Clusters: 0.4055
Silhouette Score for 5 Clusters: 0.4405
Silhouette Score for 6 Clusters: 0.4523
Silhouette Score for 7 Clusters: 0.4398
Silhouette Score for 8 Clusters: 0.4281
~~~

> Visualize data

~~~
def silhouette_plot(X, from_k, to_k):
    sil_scores=[]
    for k in range(from_k, to_k + 1):
        # Instantiate the clustering model and visualizer
        m = KMeans(n_clusters=k)
        visualizer = SilhouetteVisualizer(m)
        visualizer.fit(X) 
        visualizer.poof() 
        print(visualizer.silhouette_score_)
        sil_scores.append([visualizer.silhouette_score_, k])
    return sil_scores
    
scores=silhouette_plot(customer_df, 2, 7)
~~~

<img src="https://user-images.githubusercontent.com/67301601/143690230-f690d70b-a53b-4c0a-ab1a-0f79aa0dc41a.png"  height="300">  <img src="https://user-images.githubusercontent.com/67301601/143690232-7bbf187a-124e-4739-8589-30c67644cf1e.png"  height="300">

<img height="300" alt="Sill_4" src="https://user-images.githubusercontent.com/67301601/143690234-71fa5e1d-0b45-4010-be62-d394e793a7d0.png"> <img height="300" alt="Sill_5" src="https://user-images.githubusercontent.com/67301601/143690235-c629fa7c-0871-45a8-ab00-27cc8f0b0a71.png">

<img height="300" alt="sill_6" src="https://user-images.githubusercontent.com/67301601/143690236-28ba09e1-3166-4d00-b067-97794390306e.png"> <img height="300" alt="sill_7" src="https://user-images.githubusercontent.com/67301601/143690237-b87aa686-5073-43f9-bf64-afdc533243e2.png">

> ซึ่งจะเห็นว่าค่าเมื่อแบ่งเป็น 2 clusters ค่า Silhouette scores จะต่ำ ดังนั้นหากเลือก n_clusters ในช่วง 5, 6, 7 clusters จะให้ค่า Silhouette scores ประมาณ 0.43-0.45 ซึ่งเป็นค่าที่ค่อนข้างสูงเมื่อเปรียบเทียบกับ n_clusters อื่น ๆ 

> นอกจากนี้ เมื่อพิจารณาภาพที่ Visualize ออกมาแล้ว ดู Thickness ของแต่ละ clusters จะพบว่า n_clusters = 5 จะมีความไม่เท่ากันในแต่ละ clusters (โดยเฉพาะ cluster 0) , n_clusters = 6 มี Thickness ที่ค่อนข้างคงที่ ไม่ต่างกันมาก , n_clusters = 7 จะพบว่า clusters 2 มี Thickness ที่ไม่ค่อยเหมือนกับ clusters อื่น ๆ **จึงคิดว่า n_clusters ที่เหมาะสมที่สุดคือ 6**

----

## K-Means Clustering with 6 clusters

~~~
kmeans = KMeans(n_clusters=6).fit(customer_df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
~~~

### Plot of 3 axis (Age, Annual Income (k$), Spending Score (1-100))

<img alt="3d" src="https://user-images.githubusercontent.com/67301601/143704422-44dcf893-b9b5-426f-a3f1-d5c7809c2bef.png" height="400">

### K-Means Clustering using Age and Annual Income columns [6 clusters]

<img height="300"  src= "https://user-images.githubusercontent.com/67301601/143707506-1c728812-c6e8-43fe-95d3-d1fe7d3bb025.png"> 

> จะเห็นว่ามีการแบ่งเป็น clusters ที่ค่อนข้างมองยาก กลุ่ม cluster 5 เป็นกลุ่มที่มีการกระจายตัวของอายุ แต่ Annual Income ต่ำ ส่วน clusters อื่น ๆ ก็มีการเกาะกลุ่มกัน แต่ก็มีส่วนที่ซ้อนทับกัน เช่น cluster 0, 2 คือกลุ่มคนอายุ 20-40 และ 50-70 แต่มี Annual Income พอ ๆ กัน ส่วน cluster 1, 4 คือกลุ่มคนที่มีอายุ 30-60 และจากข้อมูลในกราฟ ก็เป็นกลุ่มคนที่มี Annual Income ที่ค่อนข้างสูง  แต่อย่างไรก็ตาม รูปนี้ก็ค่อนข้างแบ่งได้ไม่ดีนัก จุด centroid ของบาง cluster ก็ค่อนข้างใกล้กัน และมี data points ที่ซ้อนทับกันค่อนข้างเยอะ ทำให้รูปนี้อาจจะไม่สามารถนำมาเป็นข้อมูลหลักในการวิเคราะห์ marketing

### K-Means Clustering using Age and Spending Score (1-100) [6 clusters]

<img height="300" src= "https://user-images.githubusercontent.com/67301601/143718299-46c15aca-f0a5-4b25-833e-d4d8a8efeae9.png">

> จากรูปนี้สามารถแบ่งกลุ่มของลูกค้าได้ในบางส่วน คือ กลุ่ม cluster 4,5 กลุ่มพวกนี้จะไม่สามารถบอกได้แน่ชัด เพราะมีการกระจายตัวของค่าอายุ แต่ก็ชี้ให้เห็นว่า กลุ่มนี้มี Spending Score ที่ค่อนข้างต่ำ อาจจะเป็นกลุ่มลูกค้าทั่ว ๆ ไป ที่มีหลากหลายช่วงอายุ และมาซื้อสินค้าที่ mall ในระดับที่ไม่ได้เยอะและ behavior อาจจะไม่ดีซักเท่าไหร่ ทาง mall จึง assign ค่าให้ต่ำ ในส่วนของ cluster 0 ก็เป็นช่วงอายุ 20-40 ปี ที่มี score ในระดับปานกลาง เทียบเท่ากับ cluster 2  โดยใน 2 กลุ่มนี้เป็นกลุ่มที่ mall ควรให้ความสำคัญในระดับปานกลาง อาจจะหาเทคนิคการขายที่ตอบโจทย์สองกลุ่มนี้ เพราะหากนำมารวมกัน (โดยอ้างอิงจาก score ที่ใกล้เคียงกัน) ก็จะเป็นกลุ่มที่มีฐานลูกค้าจำนวนมาก  ส่วนสองกลุ่มบริเวณด้านบน คือ cluster 1, 3 เป็นกลุ่มที่ดูเหมือนจะซ้อนทับกัน แต่หากมองดี ๆ จะพบว่ามีความต่างของช่วงอายุอยู่เล็กน้อย คือ ประมาณ 20-30 ปี และ 30-40 ปี และ 2 กลุ่มนี้เป็นกลุ่มที่มี score ค่อนข้างสูง หากนำมาใช้วิเคราะห์ในด้าน marketing ก็คิดว่าทาง Mall ควรให้ความสนใจ และหาสินค้าและบริการมาตอบโจทย์ให้แก่ 2 กลุ่มนี้เป็นกรณีพิเศษ

### K-Means Clustering using Annual Income columns and Spending Score (1-100) [6 clusters]

<img height="300" src= "https://user-images.githubusercontent.com/67301601/143718865-353d682d-c651-44ea-bf34-68314b263206.png">

> การทำ clustering ด้วย data : Annual Income columns and Spending Score (1-100) จะเห็นว่ามีการแบ่งกลุ่มที่ค่อนข้างชัดเจน แต่ในรูปนี้จะติดปัญหาตรงที่ พอใช้ n_clusters = 6 กลุ่มตรงกลาง (cluster 0, 2) จะมีการแบ่งที่ดูเหมือนจะซ้อนทับกัน ดังนั้นการแบ่งเป็น 6 clusters อาจจะไม่เหมาะสมเท่าไหร่ ในกระบวนการถัดไป เลยเปลี่ยนมาแบ่งเป็น 5 clusters
                       
----

## K-Means Clustering with 5 clusters

### K-Means Clustering using Age and Annual Income columns [5 clusters]

<img height="300" src= "https://user-images.githubusercontent.com/67301601/143719113-d24bc6b4-f88c-407d-b94f-6d1288fdd7da.png">

> จะเห็นว่ามีการเปลี่ยนแปลงในบริเวณ cluster ตรงกลาง ที่มีการรวมกันจาก 2 clusters เดิม 

### K-Means Clustering using Age and Spending Score (1-100) [5 clusters]

<img height="300" src= "https://user-images.githubusercontent.com/67301601/143719172-587d5948-2a85-4233-ae29-51a87db36ed2.png">

>จะเห็นว่ามีการเปลี่ยนแปลงในบริเวณ cluster ตรงกลาง ที่มีการรวมกันจาก 2 clusters เดิม แต่ก็มีการกระจายตัวของค่าอายุที่ค่อนข้างเยอะใน cluster 4


### K-Means Clustering using Annual Income columns and Spending Score (1-100) [5 clusters]

<img height="300" src= "https://user-images.githubusercontent.com/67301601/143719231-7aeff17c-2f02-4584-9a2b-1f45ec36b8a9.png">

> จะเห็นว่าปัญหาที่บริเวณ clusters กลุ่มตรงกลาง มีการแบ่งแยกกัน ตอนนี้ได้กลับมารวมกันแล้ว ซึ่งก็แสดงให้เห็นถึงการแบ่งกลุ่มที่ชัดเจนมากยิ่งขึ้น โดยข้อมูลนี้เป็นข้อมูลที่สำคัญที่สุดในฐานข้อมูลนี้ ที่สามารถนำไปวิเคราะห์พฤติกรรมของลูกค้า และคิดกลยุทธ์ทางการตลาดที่เหมาะสมได้ **ซึ่งจะอธิบายในส่วนสรุปถัดไป**

---

## Summary

<img height="300" src= "https://user-images.githubusercontent.com/67301601/143719231-7aeff17c-2f02-4584-9a2b-1f45ec36b8a9.png">

> จากรูปที่เราเห็นจะสามารถแบ่งกลุ่มของลูกค้าได้เป็น 5 กลุ่ม ตามรายได้ต่อปี และค่า spending scores ที่กำหนดโดย Mall หากพิจารณาในแต่ละ cluster จะพบว่า

> 1. Cluster 0 คือกลุ่มลูกค้าที่มีรายได้ต่อปีที่สูง แต่มี scores ที่ค่อนข้างต่ำ ซึ่งอาจจะเดาได้ว่า สินค้าและบริการของทางร้านค้า อาจจะไม่ตอบโจทย์และไม่ถูกใจสำหรับลูกค้ากลุ่มนี้ ดังนั้น หากทาง Mall ต้องการที่จะดึงเงินที่พวกเขามี เข้ามาเป็นรายได้ของร้านค้าได้ อาจจะต้องหา Product หรือ service ที่คนกลุ่มนี้สนใจ (ถ้า Mall มีทรัพยากรมากพอที่จะสามารถทำได้) เพราะถ้าสามารถทำการตลาดกับลูกค้ากลุ่มนี้ได้ นั่นหมายความว่า เราจะมีกลุ่มลูกค้าที่มีรายได้สูงเข้ามาใช้บริการร้านค้าของเรามากขึ้น 

>  2. Cluster 1 กลุ่มนี้เป็นกลุ่มที่ตรงกันข้ามกับ cluster 0 กลุ่มนี้คือรายได้สูง และ score สูงด้วยเช่นกัน และมีจำนวนที่มากพอสมควร กลุ่มนี้ควรจัด priority มาเป็นอันดับ 1 เพราะมีกำลังซื้อสูง และ มีพฤติกรรมการซื้อของที่ร้านค้าสูงเป็นทุนเดิมอยู่แล้ว กลยุทธิ์ที่ควรใช้ คือ พยายามรักษากลุ่มลูกค้านี้ไว้ให้ได้มากที่สุด ออก promotions ที่จะดึงดูดให้พวกเขาจ่ายเงินมากกว่าเดิม และให้มาใช้บริการเป็นประจำ อาจจะใช้กลยุทธิ์ให้เป็นบัตรสมาชิก VIP เพื่อซื้อใจลูกค้าและรักษาฐานลูกค้าไว้ให้ยั่งยืน

>  3. Cluster 2 กลุ่มนี้เป็นกลุ่มที่มี Income ที่ค่อนข้างต่ำ และมี scores อยู่ในระดับค่อนข้างต่ำ อาจจะไม่ได้มี Impact ต่อทาง Mall มากนัก กลุ่มนี้อาจจะไม่ใช่กลุ่มลูกค้าของทาง Mall ก็ได้

>  4. Cluster 3 กลุ่มนี้เป็นกลุ่มที่มีรายได้ต่อปีค่อนข้างต่ำ แต่มี scores ที่ค่อนข้างสูง ถือว่าเป็นกลุ่มที่สำคัญ กลุ่มนี้เราอาจใช้กลยุทธิ์ในการคิด products, services ในราคาที่ไม่แพงมาก หรือจัด promotions ลดราคาสินค้าในบางส่วน เพื่อให้สอดคล้องกับรายได้ของผู้บริโภค และของที่ราคาถูกลง จะทำให้การซื้อสินค้าโดยลูกค้ากลุ่มนี้ มี Quantity ที่สูงขึ้น (ตามหลัก demand-supply)


>  5. Cluster 4 กลุ่มนี้เป็นกลุ่มลูกค้าที่อยู่ตรงกลาง มี Income ในระดับปานกลาง และมี spending scores ในระดับที่ปานกลาง ก็คิดว่าเป็นกลุ่มลูกค้าทั่ว ๆ ไป มีการใช้จ่ายที่สอดคล้องไปกับกำลังซื้อ การทำการตลาดก็อาจจะหาสินค้าและบริการที่มีราคาในระดับปานกลาง เพื่อให้สอดคล้องกับรายได้ของผู้บริโภค

> โดยสรุปแล้วหากต้องการที่จะ Focus ในกลุ่มลูกค้าที่สำคัญ ๆ ก็อาจจะต้อง Focus แค่ cluster 1 และ cluster 3 ที่มี spending scores ที่สูง แต่ว่าในสองกลุ่มนี้ก็มีความต่างของรายได้ต่อปีที่ค่อนข้างมาก (เฉลี่ยแล้วต่างกันประมาณ 60,000 $ (คิดจาก distance ระหว่างจุด centroids 2 จุด) คิดเป็นเงินไทยประมาณ 2,020,740 THB) ซึ่งถือว่าเป็นกลุ่มลูกค้าที่ค่อนข้างต่างกัน ดังนั้นทางร้านค้าควรหา Products, Services ที่แตกต่างกันออกไป (หากเปรียบเทียบกับสินค้าประเภทรถ ก็คือการขายรถใน segments ที่ต่างกันออกไปทั้งด้านราคาและฟังก์ชั่น) เช่น กลุ่มลูกค้าที่มีรายได้สูง ก็อาจจะเสนอขายสินค้า premium, exclusive กลุ่มที่มีรายได้น้อย ก็อาจจะเน้นสินค้าราคาถูกแต่ functional

---
> นอกจากดูข้อมูลในเรื่องของรายได้ต่อปีแล้ว ยังสามารถวิเคราะห์กลุ่มลูกค้าโดยอาศัยข้อมูลอายุได้ ดังรูปนี้ 
> 
<img height="300" src= "https://user-images.githubusercontent.com/67301601/143719172-587d5948-2a85-4233-ae29-51a87db36ed2.png">

> จากรูปนี้ข้อมูลในช่วง spending scores 0-60 จะมีหารหระจายตัวของอายุที่ค่อนข้างเยอะ ไม่สามารถบอก Insight ที่ชัดเจนได้ แต่หากดู cluster 1, 3 บริเวณด้านบน จะพบว่าข้อมูลค่อนข้างเกาะกลุ่มกันอยู่ และเป็นส่วนที่มี spending scores ที่สูง 2 กลุ่มนี้ ควรเป็นกลุ่มลูกค้าที่ทางร้านต้องให้ความสำคัญมากที่สุด ซึ่งจากรูปจะเห็นว่า 2 กลุ่มนี้มีช่วงอายุระหว่าง 20-30 ปี (วัยรุ่น-วัยทำงาน) และ 30-40 ปี (วัยทำงาน) ซึ่งทางร้านค้าควรใช้กลยุทธิ์ในการคิดค้น products , sevices หรือขายสินค้าที่ตอบโจทย์กลุ่มคนในช่วงอายุนี้เป็นหลัก เพราะจะทำให้ร้านค้าอยู่ในความสนใจของกลุ่มผู้บริโภคเหล่านี้
