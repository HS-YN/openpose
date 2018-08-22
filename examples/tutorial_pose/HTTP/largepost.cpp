/* Feel free to use this example code in any way
   you see fit (Public Domain) */

#include <sys/types.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <microhttpd.h>
#include <time.h>
//#include <darknet.h>
#include "largepost.h"
//extern network* net;

//void test_detector_http(char *datacfg, network *net, unsigned char *pFramebuffer, float thresh, float hier_thresh, char *outfile, int fullscreen, char *str_results);
static unsigned int nr_of_uploading_clients = 0;

struct connection_info_struct
{
  int connectiontype;
  struct MHD_PostProcessor *postprocessor;
  unsigned char* pFramedata;
  const char *answerstring;
  int answercode;
};

const char *askpage = "<html><body>\n\
                       Upload a file, please!<br>\n\
                       There are %u clients uploading at the moment.<br>\n\
                       <form action=\"/filepost\" method=\"post\" enctype=\"multipart/form-data\">\n\
                       <input name=\"file\" type=\"file\">\n\
                       <input type=\"submit\" value=\" Send \"></form>\n\
                       </body></html>";

const char *busypage =
  "<html><body>This server is busy, please try again later.</body></html>";

const char *completepage =
  "<html><body>The upload has been completed.</body></html>";

const char *errorpage =
  "<html><body>This doesn't seem to be right.</body></html>";
const char *servererrorpage =
  "<html><body>An internal server error has occured.</body></html>";
const char *fileexistspage =
  "<html><body>This file already exists.</body></html>";

const char *in_iteration = 
  "<html><body>It is in the iteration.</body></html>";

const char *str_request_complete =
  "<html><body>request completed.</body></html>";

char str_results[4096];

static int
send_page (struct MHD_Connection *connection, const char *page,
           int status_code)
{
  int ret;
  struct MHD_Response *response;

  response =
    MHD_create_response_from_buffer (strlen (page), (void *) page,
				     MHD_RESPMEM_MUST_COPY);
  if (!response)
    return MHD_NO;
  MHD_add_response_header (response, MHD_HTTP_HEADER_CONTENT_TYPE, "text/html");
  ret = MHD_queue_response (connection, status_code, response);
  MHD_destroy_response (response);

  return ret;
}


static int
iterate_post (void *coninfo_cls, enum MHD_ValueKind kind, const char *key,
              const char *filename, const char *content_type,
              const char *transfer_encoding, const char *data, uint64_t off,
              size_t size)
{
  struct connection_info_struct *con_info = static_cast<connection_info_struct*>(coninfo_cls);
//  FILE *fp;

  con_info->answerstring = servererrorpage;
  con_info->answercode = MHD_HTTP_INTERNAL_SERVER_ERROR;

  if (0 != strcmp (key, "application/octet-stream"))
    return MHD_NO;

  if (0 != strcmp (filename, "Framedata"))
    return MHD_NO;

  //use size and off to store all data
//  if (!con_info->fp)
//    {
//      if (NULL != (fp = fopen (filename, "filename")))
//        {
//          fclose (fp);
//          con_info->answerstring = fileexistspage;
//          con_info->answercode = MHD_HTTP_FORBIDDEN;
//          return MHD_NO;
//        }

//      con_info->fp = fopen (filename, "ab");
//      if (!con_info->fp)
//        return MHD_NO;
//    }

  if (size > 0)
    {
//      if (!fwrite (data, size, sizeof (char), con_info->fp))
//        return MHD_NO;
        con_info->answerstring = in_iteration;
        memcpy( con_info->pFramedata + off, data, size);
        if( off + size == 802816 )
        {
//          test_detector_http("cfg/voc.data", net , con_info->pFramedata, 0.24, 0.5, 0, 0, str_results);
          //3/9/2018 Chih-Yuan: return the message.
          //con_info->answerstring = str_request_complete;
          cout << "rececive a frame" << newline;
          con_info->answerstring = str_results;
        }
    }

  con_info->answercode = MHD_HTTP_OK;

  return MHD_YES;
}


//3/9/2018 Chih-Yuan : this is a callback function. It is too late the change the answerstring here.
void
request_completed (void *cls, struct MHD_Connection *connection,
                   void **con_cls, enum MHD_RequestTerminationCode toe)
{
  struct connection_info_struct *con_info = static_cast<connection_info_struct *>(*con_cls);

  if (NULL == con_info)
    return;

  if (con_info->connectiontype == POST)
    {
      if (NULL != con_info->postprocessor)
        {
          MHD_destroy_post_processor (con_info->postprocessor);
          nr_of_uploading_clients--;
        }

	  //Here should be the place I call TinyYOLO.
//	  time_t rawtime;
//	  struct tm * timeinfo;
//	  time( &rawtime);
//	  timeinfo = localtime( &rawtime);
//	  printf("Current local time and date: %s", asctime(timeinfo));
//		printf("A frame has been uploaded\n");


  		if (con_info->pFramedata)
  		{
    			free(con_info->pFramedata);			//free memory
  			con_info->pFramedata = NULL;
  		}
    }



  	if (con_info->pFramedata)
	{
    	free(con_info->pFramedata);			//free memory
		con_info->pFramedata = NULL;
	}

  free (con_info);
  *con_cls = NULL;  
}


int
answer_to_connection (void *cls, struct MHD_Connection *connection,
                      const char *url, const char *method,
                      const char *version, const char *upload_data,
                      size_t *upload_data_size, void **con_cls)
{
    
  if (NULL == *con_cls)
    {
      struct connection_info_struct *con_info;

      if (nr_of_uploading_clients >= MAXCLIENTS)
        return send_page (connection, busypage, MHD_HTTP_SERVICE_UNAVAILABLE);

      con_info = static_cast<connection_info_struct *>(malloc (sizeof (struct connection_info_struct)));
      if (NULL == con_info)
        return MHD_NO;

      //con_info->fp = NULL;
		con_info->pFramedata = NULL;		//initialize

      if (0 == strcmp (method, "POST"))
        {
          con_info->postprocessor =
            MHD_create_post_processor (connection, POSTBUFFERSIZE,
                                       iterate_post, (void *) con_info);

          if (NULL == con_info->postprocessor)
            {
              free (con_info);
              return MHD_NO;
            }

          nr_of_uploading_clients++;

          con_info->connectiontype = POST;
          con_info->answercode = MHD_HTTP_OK;
          con_info->answerstring = completepage;
		  //allocate buffer here
		  size_t datasize = 802816;

		  con_info->pFramedata = static_cast<unsigned char*>(malloc(datasize));

        }
      else
        con_info->connectiontype = GET;

      *con_cls = (void *) con_info;

      return MHD_YES;
    }

  if (0 == strcmp (method, "GET"))
    {
      char buffer[1024];

      snprintf (buffer, sizeof (buffer), askpage, nr_of_uploading_clients);
      return send_page (connection, buffer, MHD_HTTP_OK);
    }

  if (0 == strcmp (method, "POST"))
    {
      struct connection_info_struct *con_info = static_cast<connection_info_struct *>(*con_cls);

      if (0 != *upload_data_size)
        {
          MHD_post_process (con_info->postprocessor, upload_data,
                            *upload_data_size);
          *upload_data_size = 0;

          return MHD_YES;
        }
      else
	  {
	    //if (NULL != con_info->fp)
	    //  fclose (con_info->fp);		//what is the meaning of this statements? POST but upload_data_size is 0?

	    /* Now it is safe to open and inspect the file before calling send_page with a response */
	    return send_page (connection, con_info->answerstring,
			    con_info->answercode);
	  }

    }

  return send_page (connection, errorpage, MHD_HTTP_BAD_REQUEST);
}


/*int
main ()
{
  struct MHD_Daemon *daemon;
  size_t datasize = 802816;
  pFramedata = malloc(datasize);

  daemon = MHD_start_daemon (MHD_USE_SELECT_INTERNALLY, PORT, NULL, NULL,
                             &answer_to_connection, NULL,
                             MHD_OPTION_NOTIFY_COMPLETED, request_completed,
                             NULL, MHD_OPTION_END);
  if (NULL == daemon)
    return 1;
  getchar ();
  free(pFramedata);
  MHD_stop_daemon (daemon);
  return 0;
}
*/
