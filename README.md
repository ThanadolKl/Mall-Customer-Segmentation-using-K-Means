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
###
for n_cluster in [2,3,4,5,6,7,8]:
    kmeans = KMeans(n_clusters=n_cluster).fit(
        customer_df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    silhouette_avg = silhouette_score(
        customer_df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], 
        kmeans.labels_)
    
    print('Silhouette Score for %i Clusters: %0.4f' % (n_cluster, silhouette_avg))
~~~    

> Visualize data

