# Mall-Customer-Segmentation-using-K-Means
## Author : Thanadol Klainin 6S No.8


## Data Cleaning

> Clean data โดยลบ column ที่อาจจะไม่จำเป็น และเป็น Binary categories เช่น column Gender (ซึ่งจริง ๆ ก็สามารถนำมาใช้ในการวิเคราห์ได้หากสินค้าเป็นสินค้าที่แบ่งแยกเพศ แต่ในที่นี้ขอไม่ใช้ column นี้)

~~~

customer_df = df.drop(columns=['CustomerID', 'Gender'])
~~~

## Normalization

> จะเห็นว่าข้อมูล df ใน column spending scores คือข้อมูลที่ Normalize มาแล้ว และหากดู Range ของข้อมูลแต่ละ column จะพบว่า Range ไม่ได้ต่างกันมาก จึงไม่ได้มีความจำเป็นที่ต้องทำการ Normalization 

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

### kmeans clustering using Age and Annual Income columns [6 clusters]

<img height="300"  src= "https://user-images.githubusercontent.com/67301601/143707506-1c728812-c6e8-43fe-95d3-d1fe7d3bb025.png"> 

> จะเห็นว่ามีการแบ่งเป็น clusters ที่ค่อนข้างมองยาก กลุ่ม cluster 5 เป็นกลุ่มที่มีการกระจายตัวของอายุ แต่ Annual Income ต่ำ ส่วน clusters อื่น ๆ ก็มีการเกาะกลุ่มกัน แต่ก็มีส่วนที่ซ้อนทับกัน เช่น cluster 0, 2 คือกลุ่มคนอายุ 20-40 และ 50-70 แต่มี Annual Income พอ ๆ กัน ส่วน cluster 1, 4 คือกลุ่มคนที่มีอายุ 30-60 และจากข้อมูลในกราฟ ก็เป็นกลุ่มคนที่มี Annual Income ที่ค่อนข้างสูง  แต่อย่างไรก็ตาม รูปนี้ก็ค่อนข้างแบ่งได้ไม่ดีนัก จุด centroid ของบาง cluster ก็ค่อนข้างใกล้กัน และมี data points ที่ซ้อนทับกันค่อนข้างเยอะ ทำให้รูปนี้อาจจะไม่สามารถนำมาเป็นข้อมูลหลักในการวิเคราะห์ marketing

### kmeans clustering using Age and Spending Score (1-100) [6 clusters]

<img height="300" src= "https://user-images.githubusercontent.com/67301601/143718299-46c15aca-f0a5-4b25-833e-d4d8a8efeae9.png">

> จากรูปนี้สามารถแบ่งกลุ่มของลูกค้าได้ในบางส่วน คือ กลุ่ม cluster 4,5 กลุ่มพวกนี้จะไม่สามารถบอกได้แน่ชัด เพราะมีการกระจายตัวของค่าอายุ แต่ก็ชี้ให้เห็นว่า กลุ่มนี้มี Spending Score ที่ค่อนข้างต่ำ อาจจะเป็นกลุ่มลูกค้าทั่ว ๆ ไป ที่มีหลากหลายช่วงอายุ และมาซื้อสินค้าที่ mall ในระดับที่ไม่ได้เยอะและ behavior อาจจะไม่ดีซักเท่าไหร่ ทาง mall จึง assign ค่าให้ต่ำ ในส่วนของ cluster 0 ก็เป็นช่วงอายุ 20-40 ปี ที่มี score ในระดับปานกลาง เทียบเท่ากับ cluster 2  ใน 2 กลุ่มนี้เป็นกลุ่มที่ mall ควรให้ความสำคัญในระดับปานกลาง อาจจะหาเทคนิคการขายที่ตอบโจทย์สองกลุ่มนี้ เพราะหากนำมารวมกัน (โดยอ้างอิงจาก score ที่ใกล้เคียงกัน) ก็จะเป็นกลุ่มที่มีฐานลูกค้าจำนวนมาก  ส่วนสองกลุ่มบริเวณด้านบน คือ cluster 1, 3 เป็นกลุ่มที่ดูเหมือนจะซ้อนทับกัน แต่หากมองดี ๆ จะพบว่ามีความต่างของช่วงอายุอยู่เล็กน้อย คือ ประมาณ 20-30 ปี และ 30-40 ปี และ 2 กลุ่มนี้เป็นกลุ่มที่มี score ค่อนข้างสูง 
                       
