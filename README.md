# Mall-Customer-Segmentation-using-K-Means
## Author : Thanadol Klainin 6S No.8


## Data Cleaning

> Clean data โดยลบ column ที่อาจจะไม่จำเป็น และเป็น Binary categories เช่่น column Gender (ซึ่งจริง ๆ ก็สามารถนำมาใช้ในการวิเคราห์ได้หากสินค้าเป็นสินค้าที่แบ่งแยกเพศ แต่ในที่นี้ขอไม่ใช้ column นี้)

~~~

customer_df = df.drop(columns=['Gender'])

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

![Sill_2](https://user-images.githubusercontent.com/67301601/143689665-2b8a47f7-9c2a-4865-8a71-b90b65bee012.png) ![Sill_3](https://user-images.githubusercontent.com/67301601/143689686-419a804c-357b-4b14-bc86-c6557b571825.png)

