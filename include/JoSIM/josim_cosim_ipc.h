// Copyright (c) 2026 JoSIM Developers
// Licensed under the MIT License.
// Header-only C/C++ binary IPC protocol & client for josim-cosim-server (UNIX & TCP sockets).

#ifndef JOSIM_COSIM_IPC_H
#define JOSIM_COSIM_IPC_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define JOSIM_COSIM_MAGIC 0x4A43534Du /* "JCSM" */

typedef enum {
    JOSIM_IPC_OP_INIT_FILE   = 1,
    JOSIM_IPC_OP_INIT_STRING = 2,
    JOSIM_IPC_OP_STEP_TO     = 3,
    JOSIM_IPC_OP_COMMIT      = 4,
    JOSIM_IPC_OP_ROLLBACK    = 5,
    JOSIM_IPC_OP_RUN_WINDOW  = 6,
    JOSIM_IPC_OP_FINALIZE    = 7,
    JOSIM_IPC_OP_GET_TRACE   = 8,
    JOSIM_IPC_OP_GET_EVENTS  = 9,
    JOSIM_IPC_OP_SHUTDOWN    = 99
} josim_ipc_opcode_t;

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint16_t opcode;
    uint16_t flags;
    uint64_t payload_len;
} josim_ipc_req_hdr_t;

typedef struct {
    uint32_t magic;
    int32_t  status;
    uint64_t payload_len;
} josim_ipc_resp_hdr_t;
#pragma pack(pop)

#if !defined(_WIN32)

static inline int josim_ipc_write_all(int fd, const void* buf, size_t len) {
    const uint8_t* p = (const uint8_t*) buf;
    size_t         rem = len;
    while (rem > 0) {
        ssize_t n = send(fd, p, rem, 0);
        if (n < 0) {
            if (errno == EINTR) { continue; }
            return -1;
        }
        if (n == 0) { return -1; }
        p   += (size_t) n;
        rem -= (size_t) n;
    }
    return 0;
}

static inline int josim_ipc_read_all(int fd, void* buf, size_t len) {
    uint8_t* p   = (uint8_t*) buf;
    size_t   rem = len;
    while (rem > 0) {
        ssize_t n = recv(fd, p, rem, 0);
        if (n < 0) {
            if (errno == EINTR) { continue; }
            return -1;
        }
        if (n == 0) { return -1; }
        p   += (size_t) n;
        rem -= (size_t) n;
    }
    return 0;
}

/* Connect to either:
   - "unix:/path/to/sock" or "/path/to/sock"
   - "tcp:host:port" or "host:port"
   Returns socket fd >= 0 on success, -1 on error. */
static inline int josim_ipc_connect(const char* addr) {
    if (!addr || !addr[0]) { return -1; }
    const char* uPath = NULL;
    if (strncmp(addr, "unix:", 5) == 0) {
        uPath = addr + 5;
    } else if (addr[0] == '/' || (addr[0] == '.' && addr[1] == '/')) {
        uPath = addr;
    }
    if (uPath) {
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) { return -1; }
        struct sockaddr_un sun;
        memset(&sun, 0, sizeof(sun));
        sun.sun_family = AF_UNIX;
        strncpy(sun.sun_path, uPath, sizeof(sun.sun_path) - 1);
        if (connect(fd, (struct sockaddr*) &sun, sizeof(sun)) != 0) {
            close(fd);
            return -1;
        }
        return fd;
    }

    const char* tcpSpec = (strncmp(addr, "tcp:", 4) == 0) ? (addr + 4) : addr;
    const char* colon   = strrchr(tcpSpec, ':');
    if (!colon) { return -1; }
    char host[256];
    size_t hLen = (size_t) (colon - tcpSpec);
    if (hLen >= sizeof(host)) { hLen = sizeof(host) - 1; }
    memcpy(host, tcpSpec, hLen);
    host[hLen] = '\0';
    int port = atoi(colon + 1);
    if (port <= 0 || port > 65535) { return -1; }

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { return -1; }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    struct sockaddr_in sin;
    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_port   = htons((uint16_t) port);
    if (inet_pton(AF_INET, host[0] ? host : "127.0.0.1", &sin.sin_addr) <= 0) {
        struct hostent* he = gethostbyname(host);
        if (!he || !he->h_addr_list[0]) {
            close(fd);
            return -1;
        }
        memcpy(&sin.sin_addr, he->h_addr_list[0], sizeof(struct in_addr));
    }
    if (connect(fd, (struct sockaddr*) &sin, sizeof(sin)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static inline int josim_ipc_send_req(int fd, uint16_t opcode, const void* payload, uint64_t payload_len) {
    josim_ipc_req_hdr_t hdr;
    hdr.magic       = JOSIM_COSIM_MAGIC;
    hdr.opcode      = opcode;
    hdr.flags       = 0;
    hdr.payload_len = payload_len;
    if (josim_ipc_write_all(fd, &hdr, sizeof(hdr)) != 0) { return -1; }
    if (payload_len > 0 && payload) {
        if (josim_ipc_write_all(fd, payload, (size_t) payload_len) != 0) { return -1; }
    }
    return 0;
}

static inline int josim_ipc_recv_resp(int fd, int32_t* out_status, void* buf, uint64_t max_buf_len, uint64_t* out_len) {
    josim_ipc_resp_hdr_t hdr;
    if (josim_ipc_read_all(fd, &hdr, sizeof(hdr)) != 0) { return -1; }
    if (hdr.magic != JOSIM_COSIM_MAGIC) { return -1; }
    if (out_status) { *out_status = hdr.status; }
    if (out_len) { *out_len = hdr.payload_len; }
    if (hdr.payload_len > 0) {
        if (buf && hdr.payload_len <= max_buf_len) {
            if (josim_ipc_read_all(fd, buf, (size_t) hdr.payload_len) != 0) { return -1; }
        } else {
            /* Drain excess bytes */
            uint64_t rem = hdr.payload_len;
            uint8_t  tmp[256];
            while (rem > 0) {
                size_t chunk = (rem > sizeof(tmp)) ? sizeof(tmp) : (size_t) rem;
                if (josim_ipc_read_all(fd, tmp, chunk) != 0) { return -1; }
                rem -= chunk;
            }
            if (buf && hdr.payload_len > max_buf_len) { return -1; }
        }
    }
    return (hdr.status == 0) ? 0 : -1;
}

#endif /* !defined(_WIN32) */

#ifdef __cplusplus
}
#endif

#endif /* JOSIM_COSIM_IPC_H */
