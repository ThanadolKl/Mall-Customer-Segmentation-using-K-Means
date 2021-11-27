# Mall-Customer-Segmentation-using-K-Means
## Author : Thanadol Klainin 6S No.8


## Data Cleaning

> Clean data โดยลบ column ที่อาจจะไม่จำเป็น และเป็น Binary categories เช่่น column Gender (ซึ่งจริง ๆ ก็สามารถนำมาใช้ในการวิเคราห์ได้หากสินค้าเป็นสินค้าที่แบ่งแยกเพศ แต่ในที่นี้ขอไม่ใช้ column นี้)

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

> นอกจากนี้ เมื่อพิจารฯาภาพที่ Visualize ออกมาแล้ว

