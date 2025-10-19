from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Avg, Count
import requests
from django.conf import settings
from .models import Product, Review
from .serializers import ProductSerializer, ReviewSerializer

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

class ReviewViewSet(viewsets.ModelViewSet):
    queryset = Review.objects.all()
    serializer_class = ReviewSerializer

    def perform_create(self, serializer):
        review = serializer.save()
        self.analyze_review(review)

    def analyze_review(self, review):
        try:
            ai_url = f"{settings.AI_SERVICE_URL}/analyze"
            payload = {
                'review_id': review.id,
                'review_text': review.review_text,
                'rating': review.rating,
            }
            response = requests.post(ai_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                review.sentiment_score = data.get('sentiment_score')
                review.sentiment_label = data.get('sentiment_label')
                review.credibility_score = data.get('credibility_score')
                review.save()
                
        except Exception as e:
            print(f"AI analysis failed: {e}")

    @action(detail=False, methods=['get'])
    def analytics(self, request):
        analytics = {
            'total_reviews': Review.objects.count(),
            'average_rating': Review.objects.aggregate(Avg('rating'))['rating__avg'],
            'sentiment_distribution': list(Review.objects.values('sentiment_label').annotate(count=Count('id'))),
            'top_products': list(Product.objects.annotate(
                avg_rating=Avg('reviews__rating'),
                review_count=Count('reviews')
            ).filter(review_count__gt=0).order_by('-avg_rating')[:5])
        }
        return Response(analytics)