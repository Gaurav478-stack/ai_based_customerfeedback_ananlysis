from rest_framework.views import APIView
from rest_framework.response import Response
from django.db.models import Count, Avg
from reviews.models import Product, Review

class DashboardView(APIView):
    def get(self, request):
        total_products = Product.objects.count()
        total_reviews = Review.objects.count()
        
        sentiment_data = list(Review.objects.values('sentiment_label').annotate(
            count=Count('id')
        ))
        
        top_products = list(Product.objects.annotate(
            avg_rating=Avg('reviews__rating'),
            review_count=Count('reviews')
        ).filter(review_count__gt=0).order_by('-avg_rating')[:10])
        
        return Response({
            'total_products': total_products,
            'total_reviews': total_reviews,
            'sentiment_data': sentiment_data,
            'top_products': top_products,
            'average_credibility': Review.objects.aggregate(Avg('credibility_score'))['credibility_score__avg']
        })