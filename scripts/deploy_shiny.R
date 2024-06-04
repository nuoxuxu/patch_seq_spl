library(rsconnect)

# Right now shinyapps.io doesn't work because the RAM exceeds the free-tier limit
rsconnect::setAccountInfo(name='nuoxuxu',
			  token='B542F323627D8A8630BC8499CD4EF1FB',
			  secret='BkionMwHuk7XIoWRQO/+zthlbymftB+sqjg/Zgm9')
rsconnect::deployApp("/scratch/s/shreejoy/nxu/CIHR/patch_spl_app")