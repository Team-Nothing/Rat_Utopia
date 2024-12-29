export interface GenericResponse<T = never> {
  status: number;
  data: {
    code: string;
    message: string;
    data?: T;
  };
}

export interface BadResponse {
  response: GenericResponse
}
