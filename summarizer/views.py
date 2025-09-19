from django.shortcuts import render

# Create your views here.
from rest_framework import generics, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Document
from .serializers import DocumentSerializer



# List all documents of current user
class DocumentListAPIView(generics.ListCreateAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = DocumentSerializer

    def get_queryset(self):
        return Document.objects.filter(owner=self.request.user).order_by('-created_at')

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

# Retrieve / Update / Delete a document
class DocumentDetailAPIView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = DocumentSerializer
    lookup_field = 'id'

    def get_queryset(self):
        return Document.objects.filter(owner=self.request.user)
# views.py
from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Document
from .serializers import DocumentSerializer
from .utils.ml import summarize_text  # import ML function

class DocumentUploadAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        # Create document record in DB
        doc = Document.objects.create(
            owner=request.user,
            file=file,
            title=getattr(file, "name", ""),
            mimetype=getattr(file, "content_type", ""),
            size_bytes=getattr(file, "size", 0),
            status="processing"
        )

        # ========= ML Integration =========
        try:
            # Read file content as bytes
            file.seek(0)  # ensure pointer is at start
            file_bytes = file.read()
            
            # Call summarizer
            summary = summarize_text(file_bytes, file.content_type)

            # Update document with ML results
            doc.summary_text = summary
            doc.status = "ready"
            doc.save()

        except Exception as e:
            # If something goes wrong, mark document as failed
            doc.status = "failed"
            doc.last_error = str(e)
            doc.save()
            return Response(
                {"error": "ML summarization failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        # ==================================

        # Return the document info
        serializer = DocumentSerializer(doc)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
