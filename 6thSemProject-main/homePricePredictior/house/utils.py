import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from django.db.models import Case, When
from house.models import Property

def recommend_similar_properties(target_property, top_n=4, similarity_threshold=0.4):
    queryset = Property.objects.filter(is_approved=True).exclude(id=target_property.id)
    if not queryset.exists():
        return []

    df = pd.DataFrame(list(queryset.values(
        'id', 'city', 'area', 'bedrooms', 'bathrooms', 'price'
    )))

    df = pd.concat([
        pd.DataFrame([{
            'id': target_property.id,
            'city': target_property.city,
            'area': float(target_property.area),
            'bedrooms': target_property.bedrooms,
            'bathrooms': target_property.bathrooms,
            'price': float(target_property.price),
        }]),
        df
    ], ignore_index=True)

    le = LabelEncoder()
    df['city'] = le.fit_transform(df['city'])

    features = ['city', 'area', 'bedrooms', 'bathrooms', 'price']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features])

    # Optional: apply weights here if desired
    weights = [1, 0.5, 1, 1, 0.3]  # example weights
    weighted_features = scaled_features * weights

    similarity = cosine_similarity([weighted_features[0]], weighted_features[1:])[0]

    # Filter by similarity threshold
    filtered_indices = [i for i, sim in enumerate(similarity) if sim >= similarity_threshold]

    # Sort filtered by similarity descending and take top_n
    top_indices = sorted(filtered_indices, key=lambda i: similarity[i], reverse=True)[:top_n]

    recommended_ids = df.iloc[[i + 1 for i in top_indices]]['id'].tolist()

    # preserve order in queryset
    preserved_order = Case(*[When(id=pk, then=pos) for pos, pk in enumerate(recommended_ids)])
    recommended_properties = Property.objects.filter(id__in=recommended_ids).order_by(preserved_order)

    return recommended_properties