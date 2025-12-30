"""OC Serve App Entrypoint"""
from oc_serve import OCServe

oc_serve = OCServe()
application = oc_serve.get_deployment()
