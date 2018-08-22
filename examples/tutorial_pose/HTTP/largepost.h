#ifndef __LARGEPOST_H__
#define __LARGEPOST_H__

//#define PORT            8894
#define POSTBUFFERSIZE  4096
#define MAXCLIENTS      2

#define GET             0
#define POST            1

#include <microhttpd.h>
extern int answer_to_connection (void *cls, struct MHD_Connection *connection, const char *url, const char *method, const char *version, const char *upload_data, size_t *upload_data_size, void **con_cls);
extern void request_completed (void *cls, struct MHD_Connection *connection, void **con_cls, enum MHD_RequestTerminationCode toe);
extern void test_detector_http(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen);

#endif
