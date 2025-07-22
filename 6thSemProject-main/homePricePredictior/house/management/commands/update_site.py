from django.core.management.base import BaseCommand
from django.contrib.sites.models import Site

class Command(BaseCommand):
    help = 'Updates the default site domain'

    def handle(self, *args, **options):
        site, created = Site.objects.get_or_create(id=1)
        site.domain = '127.0.0.1:8000'
        site.name = 'House Price Prediction'
        site.save()
        self.stdout.write(self.style.SUCCESS('Successfully updated site domain')) 